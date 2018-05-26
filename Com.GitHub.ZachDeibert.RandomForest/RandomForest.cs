using System;
using System.Collections.Generic;
using System.Linq;

namespace Com.GitHub.ZachDeibert.RandomForest {
    [Serializable]
    public class RandomForest {
        public List<Tree> Trees;
        public int NumFeatures;
        public double Accuracy;

        public double Classify(DataPoint dataPoint) {
            return Trees.GroupBy(t => t.Classify(dataPoint)).OrderByDescending(g => g.Count()).First().Key;
        }

        RandomForest() {
        }

        public RandomForest(LearningSet data, Hyperparameters parameters) {
            Trees = new List<Tree>();
            Random rand = new Random(parameters.Seed);
            for (int i = 0; i < parameters.NumTrees; ++i) {
                Trees.Add(new Tree(data, new Random(rand.Next()), parameters));
            }
            NumFeatures = data.First().Features.Length;
            Accuracy = Trees.Average(t => t.Accuracy);
        }
    }
}
