using System;
using System.Collections.Generic;
using System.Linq;

namespace Com.GitHub.ZachDeibert.RandomForest {
    [Serializable]
    public class Node {
        public int Feature;
        public double Threshold;
        public Node Less;
        public double LessClass;
        public Node Greater;
        public double GreaterClass;

        public double Classify(DataPoint dataPoint) {
            if (dataPoint.Features[Feature] < Threshold) {
                if (Less == null) {
                    return LessClass;
                } else {
                    return Less.Classify(dataPoint);
                }
            } else if (Greater == null) {
                return GreaterClass;
            } else {
                return Greater.Classify(dataPoint);
            }
        }

        Node() {
        }

        public Node(IEnumerable<DataPoint> dataPoints, Random random, Hyperparameters parameters, int depth = 1) {
            List<int> features = new List<int>();
            while (features.Count < parameters.MaxFeatures) {
                int feature = random.Next(dataPoints.First().Features.Length);
                if (!features.Contains(feature)) {
                    features.Add(feature);
                }
            }
            double[] classes = dataPoints.GroupBy(d => d.Classification).Select(g => g.Key).ToArray();
            double bestGini = double.MaxValue;
            DataPoint[] bestLess = new DataPoint[0];
            DataPoint[] bestGreater = new DataPoint[0];
            foreach (int feature in features) {
                foreach (DataPoint dataPoint in dataPoints) {
                    DataPoint[] less = dataPoints.Where(t => t.Features[feature] < dataPoint.Features[feature]).ToArray();
                    DataPoint[] greater = dataPoints.Except(less).ToArray();
                    double gini = 0;
                    foreach (double classification in classes) {
                        foreach (DataPoint[] group in new [] { less, greater }) {
                            double amount = ((double) group.Where(d => d.Classification == classification).Count()) / (double) group.Length;
                            gini += amount * (1.0 - amount);
                        }
                    }
                    if (gini < bestGini) {
                        Feature = feature;
                        Threshold = dataPoint.Features[feature];
                        bestLess = less;
                        bestGreater = greater;
                        bestGini = gini;
                    }
                }
            }
            if (bestLess.Length == 0 || bestGreater.Length == 0) {
                LessClass = GreaterClass = dataPoints.GroupBy(d => d.Classification).OrderByDescending(g => g.Count()).First().Key;
            } else if (depth >= parameters.MaxDepth) {
                LessClass = bestLess.GroupBy(d => d.Classification).OrderByDescending(g => g.Count()).First().Key;
                GreaterClass = bestGreater.GroupBy(d => d.Classification).OrderByDescending(g => g.Count()).First().Key;
            } else {
                if (bestLess.Length > parameters.MinFeatures) {
                    Less = new Node(bestLess, random, parameters, depth + 1);
                } else {
                    LessClass = bestLess.GroupBy(d => d.Classification).OrderByDescending(g => g.Count()).First().Key;
                }
                if (bestGreater.Length > parameters.MinFeatures) {
                    Greater = new Node(bestGreater, random, parameters, depth + 1);
                } else {
                    GreaterClass = bestGreater.GroupBy(d => d.Classification).OrderByDescending(g => g.Count()).First().Key;
                }
            }
        }
    }
}
