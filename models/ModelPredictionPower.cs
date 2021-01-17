using Microsoft.ML.Data;

namespace ML.SniperSimulator
{
    public class ModelPredictionPower
    {
        [ColumnName("Score")]
        public float Power { get; set; }
    }
}
