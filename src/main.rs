use std::{error::Error, path::PathBuf};

use nalgebra::Vector2;
use tract_onnx::prelude::*;

const SIMULATION_TICKS: usize = 5;
const INPUT_YAW: f32 = 2.1;
const PITCH_MIN: f32 = -10.0;
const PITCH_MAX: f32 = 15.0;
const MAX_ROLL: f32 = 45.0;
const EMERGENCY_PITCH_LIMIT: f32 = -2.0;

type InferenceResult = Result<(f32, f32), Box<dyn Error>>;

#[derive(Debug, Clone, Copy)]
pub struct AircraftDynamics {
    pub pitch: f32,
    pub roll: f32,
    pub altitude: f32,
    pub airspeed: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AiAdaCorrection {
    pub safe_to_execute: bool,
    pub corrected_pitch: f32,
    pub warning_level: u8,
}

#[derive(Debug)]
pub struct DeaLearningEngine {
    pub total_samples: usize,
    pub sensitivity_coefficient: f32,
}

impl DeaLearningEngine {
    pub fn new() -> Self {
        Self {
            total_samples: 1_250,
            sensitivity_coefficient: 1.25,
        }
    }

    pub fn feedback_refinement(&mut self, ai_intervened: bool) {
        self.total_samples = self.total_samples.saturating_add(1);
        if ai_intervened {
            self.sensitivity_coefficient += 0.002;
            println!(
                "[DEA MATRIX FEEDBACK] Risk pattern verified. Sensitivity scaled to {:.4}",
                self.sensitivity_coefficient
            );
        }
    }
}

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ai_ada_model.onnx")
}

fn is_unsafe(plane: &AircraftDynamics) -> bool {
    plane.pitch < PITCH_MIN || plane.pitch > PITCH_MAX || plane.roll.abs() > MAX_ROLL
}

fn rule_based_inference(plane: &AircraftDynamics) -> InferenceResult {
    let unsafe_state = is_unsafe(plane);
    let corrected_pitch = if unsafe_state {
        (EMERGENCY_PITCH_LIMIT - plane.pitch) * 0.3
    } else {
        plane.pitch
    };
    Ok((corrected_pitch, if unsafe_state { 0.0 } else { 1.0 }))
}

fn run_simulation<F>(mut infer: F) -> Result<(), Box<dyn Error>>
where
    F: FnMut(&AircraftDynamics) -> InferenceResult,
{
    let mut dea_engine = DeaLearningEngine::new();
    let mut plane = AircraftDynamics {
        pitch: -15.5,
        roll: 45.0,
        altitude: 10_000.0,
        airspeed: 280.0,
    };

    println!("--- [Starting Flight Simulation Loop ({SIMULATION_TICKS} Ticks)] ---");

    for tick in 1..=SIMULATION_TICKS {
        println!("\n[Flight Tick #{tick}]");
        println!(
            "-> Current Plane State: Pitch {:.1}°, Roll {:.1}°, Alt {:.0}ft",
            plane.pitch, plane.roll, plane.altitude
        );

        let (raw_pitch, raw_safety_score) = infer(&plane)?;
        let scaled_output =
            Vector2::new(raw_pitch, raw_safety_score) * dea_engine.sensitivity_coefficient;
        let safe_to_execute = scaled_output.y > 0.5;
        let correction = AiAdaCorrection {
            safe_to_execute,
            corrected_pitch: scaled_output.x,
            warning_level: if safe_to_execute { 0 } else { 2 },
        };

        if !correction.safe_to_execute {
            println!(
                "[EMERGENCY OVERWRITE] AI-Ada correction active. Warning Level: {}",
                correction.warning_level
            );
            plane.pitch =
                (plane.pitch + correction.corrected_pitch.abs() * 0.3).min(EMERGENCY_PITCH_LIMIT);
            plane.roll *= 0.7;
            plane.altitude -= 50.0;
        } else {
            println!("[STABLE] Pilot inputs within safe envelope.");
            plane.pitch += 1.0;
        }

        dea_engine.feedback_refinement(!correction.safe_to_execute);
    }

    println!("\n--- [Simulation Completed Successfully] ---");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== AI-Ada Flight Control & Simulation Core Initializing ===");

    let model_path = model_path();
    if !model_path.is_file() {
        println!(
            "[WARN] Model file not found at {}. Running rule-based fallback.",
            model_path.display()
        );
        return run_simulation(rule_based_inference);
    }

    let model = tract_onnx::onnx()
        .model_for_path(&model_path)?
        .into_optimized()?
        .into_runnable()?;
    println!("AI-Ada ONNX Neural Core loaded successfully.\n");

    run_simulation(|plane| {
        let input = tract_ndarray::arr2(&[[plane.pitch, plane.roll, INPUT_YAW, plane.airspeed]])
            .into_tensor();
        let outputs = model.run(tvec!(input.into()))?;
        let values = outputs
            .first()
            .ok_or("ONNX model returned no output tensor")?
            .to_array_view::<f32>()?;

        if values.shape() != [1, 2] {
            return Err(format!(
                "expected ONNX output shape [1, 2], got {:?}",
                values.shape()
            )
            .into());
        }

        let corrected_pitch = values[[0, 0]];
        let safety_score = values[[0, 1]];
        if !corrected_pitch.is_finite() || !safety_score.is_finite() {
            return Err("ONNX model returned a non-finite decision".into());
        }
        Ok((corrected_pitch, safety_score))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsafe_envelope_checks_pitch_and_roll_limits() {
        let mut plane = AircraftDynamics {
            pitch: 0.0,
            roll: 0.0,
            altitude: 10_000.0,
            airspeed: 280.0,
        };
        assert!(!is_unsafe(&plane));

        plane.pitch = PITCH_MIN - 0.1;
        assert!(is_unsafe(&plane));
        plane.pitch = PITCH_MAX + 0.1;
        assert!(is_unsafe(&plane));
        plane.pitch = 0.0;
        plane.roll = MAX_ROLL + 0.1;
        assert!(is_unsafe(&plane));
    }

    #[test]
    fn feedback_counts_every_sample_and_increases_sensitivity_on_intervention() {
        let mut engine = DeaLearningEngine::new();
        engine.feedback_refinement(false);
        assert_eq!(engine.total_samples, 1_251);
        assert_eq!(engine.sensitivity_coefficient, 1.25);

        engine.feedback_refinement(true);
        assert_eq!(engine.total_samples, 1_252);
        assert!((engine.sensitivity_coefficient - 1.252).abs() < f32::EPSILON);
    }

    #[test]
    fn rule_based_fallback_flags_unsafe_states() {
        let unsafe_plane = AircraftDynamics {
            pitch: -15.5,
            roll: 45.0,
            altitude: 10_000.0,
            airspeed: 280.0,
        };
        let (corrected_pitch, safety_score) =
            rule_based_inference(&unsafe_plane).expect("fallback inference should succeed");

        assert!(corrected_pitch > 0.0);
        assert_eq!(safety_score, 0.0);
    }
}
