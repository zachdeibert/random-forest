using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Com.GitHub.ZachDeibert.RandomForest {
    public class LearningSet : List<DataPoint> {
        static DataPoint ParseRow(string line) {
            double[] arr = line.Split(',').Select(s => double.Parse(s)).ToArray();
            return new DataPoint {
                Features = arr.Take(arr.Length - 1).ToArray(),
                Classification = arr[arr.Length - 1]
            };
        }

        public LearningSet(string filename) {
            string[] lines = File.ReadAllLines(filename);
            try {
                Add(ParseRow(lines[0]));
            } catch {
                // It could be a header, in which case it won't parse.
            }
            AddRange(lines.Skip(1).Where(l => l.Length > 0).Select(l => ParseRow(l)));
            if (this.GroupBy(d => d.Features.Length).Count() != 1) {
                throw new InvalidDataException("All rows must have the same number of columns");
            }
        }
    }
}
