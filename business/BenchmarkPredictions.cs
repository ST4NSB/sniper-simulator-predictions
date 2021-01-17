using Microsoft.ML;
using System;

namespace ML.SniperSimulator
{
    public class BenchmarkPredictions
    {
        public string _input;
        private MLContext _ml;
        private ITransformer _model;
        private IDataView _testData;

        public BenchmarkPredictions(string input)
        {
            this._input = input;
            var random = new Random();
            this._ml = new MLContext(seed: random.Next());
        }

        public void TrainModel(float testFraction, string predictionAttribute, TrainerType trainerType, int maximumNumberOfIterations = 5)
        {
            var data = _ml.Data.LoadFromTextFile<ModelInput>(_input, separatorChar: ',', hasHeader: true);
            var dataSplit = _ml.Data.TrainTestSplit(data, testFraction: testFraction);
            var trainData = dataSplit.TrainSet;
            _testData = dataSplit.TestSet;

            switch (trainerType)
            {
                case TrainerType.LbfgsPoissonRegression:
                    var pipelinePoisson = _ml.Transforms.Categorical.OneHotEncoding(outputColumnName: "BenchmarkNameEncoded", inputColumnName: "BenchmarkName")
                        .Append(_ml.Transforms.Categorical.OneHotEncoding(outputColumnName: "CoresNumberEncoded", inputColumnName: "CoresNumber"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "IPC"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "Area"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "Instructions"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "Cycles"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "Power"))
                        .Append(_ml.Transforms.Concatenate("Features", "IPC", "BenchmarkNameEncoded", "CoresNumberEncoded",
                                                            "Power", "Area", "Instructions", "Cycles"))
                        .Append(_ml.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: predictionAttribute))
                        .Append(_ml.Regression.Trainers.LbfgsPoissonRegression());

                    _model = pipelinePoisson.Fit(trainData);
                    break;
                case TrainerType.Sdca:
                    var pipelineSCDA = _ml.Transforms.Categorical.OneHotEncoding(outputColumnName: "BenchmarkNameEncoded", inputColumnName: "BenchmarkName")
                        .Append(_ml.Transforms.Categorical.OneHotEncoding(outputColumnName: "CoresNumberEncoded", inputColumnName: "CoresNumber"))
                        .Append(_ml.Transforms.NormalizeMinMax(outputColumnName: "IPC"))
                        .Append(_ml.Transforms.NormalizeMinMax(outputColumnName: "Area"))
                        .Append(_ml.Transforms.NormalizeMinMax(outputColumnName: "Instructions"))
                        .Append(_ml.Transforms.NormalizeMinMax(outputColumnName: "Cycles"))
                        .Append(_ml.Transforms.NormalizeMinMax(outputColumnName: "Power"))
                        .Append(_ml.Transforms.Concatenate("Features", "IPC", "BenchmarkNameEncoded", "CoresNumberEncoded",
                                                            "Power", "Area", "Instructions", "Cycles"))
                        .Append(_ml.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: predictionAttribute))
                        .Append(_ml.Regression.Trainers.Sdca(maximumNumberOfIterations: maximumNumberOfIterations));

                    _model = pipelineSCDA.Fit(trainData);
                    break;
                case TrainerType.OnlineGradientDescent:
                    var pipelineOGD = _ml.Transforms.Categorical.OneHotEncoding(outputColumnName: "BenchmarkNameEncoded", inputColumnName: "BenchmarkName")
                        .Append(_ml.Transforms.Categorical.OneHotEncoding(outputColumnName: "CoresNumberEncoded", inputColumnName: "CoresNumber"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "IPC"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "Area"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "Instructions"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "Cycles"))
                        .Append(_ml.Transforms.NormalizeMeanVariance(outputColumnName: "Power"))
                        .Append(_ml.Transforms.Concatenate("Features", "IPC", "BenchmarkNameEncoded", "CoresNumberEncoded",
                                                            "Power", "Area", "Instructions", "Cycles"))
                        .Append(_ml.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: predictionAttribute))
                        .Append(_ml.Regression.Trainers.OnlineGradientDescent());

                    _model = pipelineOGD.Fit(trainData);
                    break;
            }
        }

        public string EvaluateAndShowResults()
        {
            var predictions = _model.Transform(_testData);
            var metrics = _ml.Regression.Evaluate(predictions, "Label", "Score");

            var returnable = "";
            returnable += ($"*************************************************\n");
            returnable += ($"*       Model quality metrics evaluation         \n");
            returnable += ($"*------------------------------------------------\n");
            returnable += ($"*       RSquared Score:      {metrics.RSquared}\n");
            returnable += ($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError}\n");
            returnable += ($"*************************************************\n");
            return returnable;
        }

        public string TestSinglePredictionIPC(ModelInput inputTestData, float expected)
        {
            var predictionFunction = _ml.Model.CreatePredictionEngine<ModelInput, ModelPredictionIpc>(_model);
            var prediction = predictionFunction.Predict(inputTestData);

            var returnable = "";
            returnable += ($"**********************************************************************\n");
            returnable += ($"Predicted value: {prediction.IPC}, actual value: {expected}\n");
            returnable += ($"**********************************************************************\n");
            return returnable;
        }

        public string TestSinglePredictionPower(ModelInput inputTestData, float expected)
        {
            var predictionFunction = _ml.Model.CreatePredictionEngine<ModelInput, ModelPredictionPower>(_model);
            var prediction = predictionFunction.Predict(inputTestData);

            var returnable = "";
            returnable += ($"**********************************************************************\n");
            returnable += ($"Predicted value: {prediction.Power}, actual value: {expected}\n");
            returnable += ($"**********************************************************************\n");
            return returnable;
        }
    }
}
