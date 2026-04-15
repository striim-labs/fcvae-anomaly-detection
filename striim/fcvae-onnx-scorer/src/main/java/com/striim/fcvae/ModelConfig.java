package com.striim.fcvae;

/**
 * POJO for model_config.json, parsed via Gson.
 */
public class ModelConfig {
    public String model_name;
    public OnnxConfig onnx;
    public ScalerConfig scaler;
    public ThresholdConfig thresholds;
    public ScorerStats scorer;

    public static class OnnxConfig {
        public int opset_version;
        public int window_size;
        public String input_name;
        public String output_name;
    }

    public static class ScalerConfig {
        public String type;
        public double mean;
        public double scale;
    }

    public static class ThresholdConfig {
        public double last_point_threshold;
        public double point_threshold;
        public double window_threshold;
    }

    public static class ScorerStats {
        public double normal_score_mean;
        public double normal_score_std;
    }
}
