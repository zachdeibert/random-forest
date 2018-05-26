using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Com.GitHub.ZachDeibert.RandomForest {
    [Serializable]
    public class RandomForest {
        public Tree[] Trees;
        public int NumFeatures;
        public double Accuracy;

        public double Classify(DataPoint dataPoint) {
            return Trees.GroupBy(t => t.Classify(dataPoint)).OrderByDescending(g => g.Count()).First().Key;
        }

        RandomForest() {
        }

        public RandomForest(LearningSet data, Hyperparameters parameters) {
            List<Task<Tree>> tasks = new List<Task<Tree>>();
            Random rand = new Random(parameters.Seed);
            for (int i = 0; i < parameters.NumTrees; ++i) {
                Random treeRandom = new Random(rand.Next());
                tasks.Add(Task.Run(() => new Tree(data, treeRandom, parameters)));
            }
            NumFeatures = data.First().Features.Length;
            Task.WaitAll(tasks.ToArray());
            Trees = tasks.Select(t => t.Result).ToArray();
            Accuracy = Trees.Average(t => t.Accuracy);
        }
    }
}
