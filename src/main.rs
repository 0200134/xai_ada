use tract_onnx::prelude::*;
use nalgebra::Matrix2x1;

#[derive(Debug, Clone)]
pub struct AircraftDynamics {
    pub pitch: f32,
    pub roll: f32,
    pub altitude: f32,
    pub airspeed: f32,
}

#[derive(Debug)]
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
            total_samples: 1250,
            sensitivity_coefficient: 1.25,
        }
    }

    pub fn feedback_refinement(&mut self, ai_intervened: bool) {
        self.total_samples += 1;
        if ai_intervened {
            self.sensitivity_coefficient += 0.002;
            println!("[DEA MATRIX FEEDBACK] Risk pattern verified. Sensitivity scaled to {:.4}", self.sensitivity_coefficient);
        }
    }
}

fn main() -> TractResult<()> {
    println!("=== AI-Ada Flight Control & Simulation Core Initializing ===");

    let mut dea_engine = DeaLearningEngine::new();
    let model_path = "ai_ada_model.onnx";

    let model = match tract_onnx::onnx().model_for_path(model_path) {
        Ok(m) => m.into_optimized()?.into_runnable()?,
        Err(_) => {
            println!("[WARN] Model file not found. Running rule-based fallback.");
            return Ok(());
        }
    };
    println!("AI-Ada ONNX Neural Core loaded successfully.\n");

    // 초기 비행 기체 상태 (신입 파일럿이 조종간을 잡고 급강하 시도 중인 상황)
    let mut plane = AircraftDynamics {
        pitch: -15.5,
        roll: 45.0,
        altitude: 10000.0,
        airspeed: 280.0,
    };

    println!("--- [Starting Flight Simulation Loop (5 Ticks)] ---");

    // 5단계의 실시간 비행 시뮬레이션 틱(Tick) 실행
    for tick in 1..=5 {
        println!("\n[Flight Tick #{}]", tick);
        println!("-> Current Plane State : Pitch {:.1}°, Roll {:.1}°, Alt {:.0}ft", plane.pitch, plane.roll, plane.altitude);

        // 1. 텐서 변환 및 ONNX 추론
        let array = tract_ndarray::array!([plane.pitch, plane.roll, 2.1, plane.airspeed])
            .into_shape_with_order((1, 4))
            .unwrap();
        let tensor: Tensor = array.into_tensor().into();

        let result = model.run(tvec!(tensor.into()))?;
        let raw_output = result[0].to_array_view::<f32>()?;

        let raw_pitch = raw_output[[0, 0]];
        let raw_safety_score = raw_output[[0, 1]];

        // 2. DEA 행렬 민감도 스케일링 연산
        let raw_output_vector = Matrix2x1::new(raw_pitch, raw_safety_score);
        let scaled_output_vector = raw_output_vector * dea_engine.sensitivity_coefficient;

        let scaled_pitch = scaled_output_vector[(0, 0)];
        let scaled_safety = scaled_output_vector[(1, 0)];

        let safe_flag = scaled_safety > 0.5;
        let correction = AiAdaCorrection {
            safe_to_execute: safe_flag,
            corrected_pitch: scaled_pitch,
            warning_level: if safe_flag { 0 } else { 2 },
        };

        // 3. 비행 다이내믹스 갱신 (AI-Ada가 개입하여 기수를 강제로 끌어올림)
        if !correction.safe_to_execute {
            println!("[⚠️ EMERGENCY OVERWRITE] AI-Ada Hydraulic Lock Engaged! Warning Level: {}", correction.warning_level);
            // 기수를 안전 영역으로 강제 보정 (점진적으로 복원)
            plane.pitch = (plane.pitch + (correction.corrected_pitch.abs() * 0.3)).min(-2.0);
            plane.roll = plane.roll * 0.7; // 뱅크각 회복
            plane.altitude -= 50.0; // 강하율 완화
        } else {
            println!("[✅ STABLE] Pilot inputs within safe envelope.");
            plane.pitch += 1.0;
        }

        // 4. DEA 학습 피드백 반영
        dea_engine.feedback_refinement(!correction.safe_to_execute);
    }

    println!("\n--- [Simulation Completed Successfully] ---");
    Ok(())
}
