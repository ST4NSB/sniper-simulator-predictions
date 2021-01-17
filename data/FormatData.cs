using CsvHelper;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;


namespace ML.SniperSimulator
{
    public static class FormatData
    {
        private static string DeleteBeforeEscapeChar(this string input)
        {
            return input.Split('\\').Last();
        }

        private static List<float> ReadStringFromFile(string dir, string fileName,  string filter)
        {
            string line;
            var path = Path.Combine(dir, fileName);
            var values = new List<float>();

            if (File.Exists(path))
            {
                var details = new StreamReader(path);
                while ((line = details.ReadLine()) != null)
                {
                    if (line.Contains(filter))
                    {
                        var matchString = Regex.Matches(line, @"[-+]?[0-9]*\.?[0-9]+");
                        foreach (Match match in matchString)
                        {
                            var matchValue = float.Parse(match.Value, CultureInfo.InvariantCulture.NumberFormat);
                            values.Add(matchValue);
                        }
                        
                        return values;
                    }
                }
            }

            return values;
        }

        public static string FormatBenchmarksToCsv(string fileName)
        {
            var currDir = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), @"..\..\..\files\"));
            var benchmarks = Path.Combine(currDir, "runs_small");

            if(File.Exists(Path.Combine(currDir, fileName)))
            {
                return Path.Combine(currDir, fileName);
            }

            var modelsList = new List<ModelInput>();
            var subDirCores = Directory.GetDirectories(benchmarks);

            foreach(var dirCore in subDirCores)
            {
                var subDirBench = Directory.GetDirectories(dirCore);

                foreach (var dirBench in subDirBench)
                {
                    var nrIpcs = ReadStringFromFile(dirBench, "sim.out", "IPC");
                    var nrInstr = ReadStringFromFile(dirBench, "sim.out", "Instructions");
                    var nrCycles = ReadStringFromFile(dirBench, "sim.out", "Cycles");

                    for (int i = 0; i < nrIpcs.Count; i++) 
                    {
                        var model = new ModelInput();
                        model.BenchmarkName = dirBench.DeleteBeforeEscapeChar();
                        model.CoresNumber = dirCore.DeleteBeforeEscapeChar();
                        model.Area = ReadStringFromFile(dirBench, "power.txt", "Area").FirstOrDefault();
                        model.Instructions = (int)nrInstr[i];
                        model.Cycles = (int)nrCycles[i];
                        model.IPC = nrIpcs[i];
                        model.Power = ReadStringFromFile(dirBench, "power.txt", "Peak Power").FirstOrDefault();
                        modelsList.Add(model);
                    }
                }
            }

            using (var mem = new MemoryStream())
            using (var writer = new StreamWriter(mem))
            using (var csvWriter = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csvWriter.Configuration.Delimiter = ",";
                csvWriter.Configuration.HasHeaderRecord = true;
                csvWriter.Configuration.AutoMap<ModelInput>();

                csvWriter.WriteRecords(modelsList);

                writer.Flush();
                var result = Encoding.UTF8.GetString(mem.ToArray());

                File.WriteAllText(Path.Combine(currDir, fileName), result);
            }

            return Path.Combine(currDir, fileName);
        }
    }
}
