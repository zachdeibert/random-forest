using System;
using System.Collections.Generic;
using System.Linq;

namespace Com.GitHub.ZachDeibert.RandomForest {
    [Serializable]
    public class Tree {
        public Node Root;
        public double Accuracy;

        public double Classify(DataPoint dataPoint) {
            return Root.Classify(dataPoint);
        }

        Tree() {
        }

        public Tree(LearningSet data, Random random, Hyperparameters parameters) {
            List<DataPoint> bag = new List<DataPoint>();
            List<DataPoint> outOfBag = new List<DataPoint>();
            foreach (DataPoint point in data) {
                if (random.NextDouble() < parameters.OutOfBag) {
                    outOfBag.Add(point);
                } else {
                    bag.Add(point);
                }
            }
            Root = new Node(bag, random, parameters);
            Accuracy = outOfBag.Average(d => Root.Classify(d) == d.Classification ? 1.0 : 0.0);
        }
    }
}
