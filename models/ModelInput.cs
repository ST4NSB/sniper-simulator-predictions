using Microsoft.ML.Data;

namespace ML.SniperSimulator
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public string BenchmarkName { get; set; }
        [LoadColumn(1)]
        public string CoresNumber { get; set; }
        [LoadColumn(2)]
        public float Area { get; set; }
        [LoadColumn(3)]
        public float Instructions { get; set; }
        [LoadColumn(4)]
        public float Cycles { get; set; }
        [LoadColumn(5)]
        public float IPC { get; set; } 
        [LoadColumn(6)]
        public float Power { get; set; }
    }
}