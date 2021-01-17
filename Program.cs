using System;

namespace ML.SniperSimulator
{
    public class Program
    {
        static void runIPC(BenchmarkPredictions predictionModel)
        {
            predictionModel.TrainModel(testFraction: 0.25f, "IPC", trainerType: TrainerType.Sdca, maximumNumberOfIterations: 100);
            var results = predictionModel.EvaluateAndShowResults();
            Console.WriteLine(results);
        }

        static void runPower(BenchmarkPredictions predictionModel)
        {
            predictionModel.TrainModel(testFraction: 0.25f, "Power", trainerType: TrainerType.Sdca, maximumNumberOfIterations: 100);
            var results = predictionModel.EvaluateAndShowResults();
            Console.WriteLine(results);
        }

        static void Main(string[] args)
        {
            var csvFileName = "sniper_input_data.csv";
            var input = FormatData.FormatBenchmarksToCsv(csvFileName);

            var predictionModel = new BenchmarkPredictions(input);
            runIPC(predictionModel);
            runPower(predictionModel);

            var testSampleIPC = new ModelInput
            {
                BenchmarkName = "barnes",
                CoresNumber = "cores_8",
                Area = 388.384f,
                Instructions = 211795259f,
                Cycles = 258886957f,
                Power = 256.679f,
            };

            var resultIPC = predictionModel.TestSinglePredictionIPC(testSampleIPC, expected: 0.85f);
            Console.WriteLine(resultIPC);
        }
    }
}
