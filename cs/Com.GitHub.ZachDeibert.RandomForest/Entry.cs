using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Xml.Serialization;

namespace Com.GitHub.ZachDeibert.RandomForest {
    public static class Entry {
        static void InvalidArgs() {
            Console.WriteLine("Usage: {0} [options] [training data.csv] [trained model.bin] [trained model.xml] [<feature 1> <feature 2> ...]", Path.GetFileName(Process.GetCurrentProcess().MainModule.FileName));
            Console.WriteLine();
            Console.WriteLine("If a .csv file is given, the model will be trained from that data.");
            Console.WriteLine("If a .bin or .xml file is given with a .csv file, the model will be stored in that file.");
            Console.WriteLine("If a .bin or .xml file is given without a .csv file, the model will be read from that file.");
            Console.WriteLine("If a list of features is given, the classification result will be outputted.");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("    --num-trees    Sets the number of trees to generate in the forest (def=10)");
            Console.WriteLine("    --max-features Sets the maxinum number of features to include in a tree (def=sqrt(total features))");
            Console.WriteLine("    --min-features Sets the minimum number of data points to allow each feature to separate in the decision tree (def=1)");
            Console.WriteLine("    --max-depth    Sets the maxinum depth of the decision trees (def=10)");
            Console.WriteLine("    --seed         Sets the PRNG seed for reproducible results (def=random)");
            Console.WriteLine("    --oob          Sets the proportion of training data to keep out of the bag (def=0.3)");
            Environment.Exit(0);
        }

        public static void Main(string[] args) {
            Hyperparameters parameters = new Hyperparameters {
                NumTrees = 10,
                MaxFeatures = -1,
                MinFeatures = 1,
                MaxDepth = 10,
                Seed = (int) (DateTime.Now.Ticks % int.MaxValue),
                OutOfBag = 0.3
            };
            string trainingFile = null;
            string serializedFile = null;
            List<double> testData = new List<double>();
            for (int i = 0; i < args.Length; ++i) {
                switch (args[i]) {
                    case "--num-trees":
                        if (i + 1 >= args.Length) {
                            InvalidArgs();
                        }
                        if (!int.TryParse(args[++i], out parameters.NumTrees)) {
                            InvalidArgs();
                        }
                        break;
                    case "--max-features":
                        if (i + 1 >= args.Length) {
                            InvalidArgs();
                        }
                        if (!int.TryParse(args[++i], out parameters.MaxFeatures)) {
                            InvalidArgs();
                        }
                        break;
                    case "--min-features":
                        if (i + 1 >= args.Length) {
                            InvalidArgs();
                        }
                        if (!int.TryParse(args[++i], out parameters.MinFeatures)) {
                            InvalidArgs();
                        }
                        break;
                    case "--max-depth":
                        if (i + 1 >= args.Length) {
                            InvalidArgs();
                        }
                        if (!int.TryParse(args[++i], out parameters.MaxDepth)) {
                            InvalidArgs();
                        }
                        break;
                    case "--seed":
                        if (i + 1 >= args.Length) {
                            InvalidArgs();
                        }
                        if (!int.TryParse(args[++i], out parameters.Seed)) {
                            InvalidArgs();
                        }
                        break;
                    case "--oob":
                        if (i + 1 >= args.Length) {
                            InvalidArgs();
                        }
                        if (!double.TryParse(args[++i], out parameters.OutOfBag)) {
                            InvalidArgs();
                        }
                        break;
                    default:
                        double val;
                        if (args[i].EndsWith(".csv")) {
                            if (trainingFile == null) {
                                trainingFile = args[i];
                            } else {
                                InvalidArgs();
                            }
                        } else if (args[i].EndsWith(".bin") || args[i].EndsWith(".xml")) {
                            if (serializedFile == null) {
                                serializedFile = args[i];
                            } else {
                                InvalidArgs();
                            }
                        } else if (double.TryParse(args[i], out val)) {
                            testData.Add(val);
                        } else {
                            InvalidArgs();
                        }
                        break;
                }
            }
            RandomForest forest = null;
            if (trainingFile == null) {
                if (serializedFile == null || !File.Exists(serializedFile)) {
                    Console.WriteLine("No model source");
                    InvalidArgs();
                } else {
                    if (serializedFile.EndsWith(".xml")) {
                        XmlSerializer serializer = new XmlSerializerFactory().CreateSerializer(typeof(RandomForest));
                        using (Stream stream = new FileStream(serializedFile, FileMode.Open, FileAccess.Read)) {
                            forest = (RandomForest) serializer.Deserialize(stream);
                        }
                    } else {
                        BinaryFormatter serializer = new BinaryFormatter();
                        using (Stream stream = new FileStream(serializedFile, FileMode.Open, FileAccess.Read)) {
                            forest = (RandomForest) serializer.Deserialize(stream);
                        }
                    }
                }
            } else {
                LearningSet learningSet = new LearningSet(trainingFile);
                if (parameters.MaxFeatures == -1) {
                    parameters.MaxFeatures = (int) Math.Sqrt(learningSet.First().Features.Length);
                }
                forest = new RandomForest(learningSet, parameters);
                if (serializedFile != null) {
                    if (serializedFile.EndsWith(".xml")) {
                        XmlSerializer serializer = new XmlSerializerFactory().CreateSerializer(typeof(RandomForest));
                        using (Stream stream = new FileStream(serializedFile, FileMode.Create, FileAccess.Write)) {
                            serializer.Serialize(stream, forest);
                        }
                    } else {
                        BinaryFormatter serializer = new BinaryFormatter();
                        using (Stream stream = new FileStream(serializedFile, FileMode.Create, FileAccess.Write)) {
                            serializer.Serialize(stream, forest);
                        }
                    }
                }
            }
            if (testData.Count > 0) {
                if (testData.Count == forest.NumFeatures) {
                    Console.WriteLine(forest.Classify(new DataPoint {
                        Features = testData.ToArray()
                    }));
                } else {
                    Console.WriteLine("Invalid number of features");
                    InvalidArgs();
                }
            } else {
                Console.WriteLine("Accuracy: {0}%", forest.Accuracy * 100.0);
            }
        }
    }
}
