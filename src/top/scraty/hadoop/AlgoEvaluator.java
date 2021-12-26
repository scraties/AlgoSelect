package top.scraty.hadoop;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class AlgoEvaluator {
    public static void evaluate(Classifier classifier, String[] test, Instances testFeatures) throws Exception {
        double optimality = 0.0;
        double accuracy = 0.0;
        for(int i = 0; i < test.length; i++) {
            double opt = getOptimality(classifier, test[i], testFeatures.instance(i));
            optimality += opt;
            if(opt > 0.9999) {
                accuracy += 1.0;
            }
        }

        System.out.println();
        System.out.println("EVALUATE RESULT:");
        System.out.println("evaluateOptimality: " + optimality / test.length);
        System.out.println("evaluateAccuracy: " + accuracy / test.length);
    }
    private static double getOptimality(Classifier classifier, String datasetName, Instance feature) throws Exception {
        System.out.println("evaluate: " + datasetName);
        int num = (int)classifier.classifyInstance(feature);
        System.out.println("RandomForest Result: " + DatasetFeature.classifier[num]);

        try {
            double acc = DatasetFeature.getClassifierAcc(DatasetFeature.classifier[num], datasetName, DatasetFeature.loadDataset(datasetName));
            System.out.println("result acc: " + acc);
            System.out.println("max acc: " + feature.classValue());
            return acc / feature.classValue();
        } catch (Exception e) {
            System.out.println(DatasetFeature.classifier[num] + " on " + datasetName + "\t\t FAILED ");
            return 0.0;
        }
    }
}
