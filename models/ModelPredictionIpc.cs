using Microsoft.ML.Data;

namespace ML.SniperSimulator
{
    public class ModelPredictionIpc
    {
        [ColumnName("Score")]
        public float IPC { get; set; }
    }
}
