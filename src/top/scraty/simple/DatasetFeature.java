package top.scraty.simple;

import evaluation.evaluators.Evaluator;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.ClassifierLists;
import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.RandomProjection;

import java.io.IOException;
import java.util.ArrayList;

public class DatasetFeature {
    public final static String[] classifier = {"FastShapelets", "DD_DTW", "DTD_C", "BOSS", "cBOSS", "WEASEL", "NN_CID", "SAX_1NN",
            "SpatialBOSS", "ProximityForest", "TSF", "SAXVSM", "SpatialBOSS", "SAX_1NN", "LPS"};
    public final static int seed = 19260817;

    public static Instances getFeatures(String[] dataset, boolean isTrain) {
        Instances features = initFeatures();

        for(String datasetName : dataset) {
            try {
                Instance feature = new DenseInstance(1.0, getDatasetFeature(datasetName, isTrain));
                feature.setDataset(features);
                features.add(feature);
                System.out.println(feature);
            } catch (Exception e) {
                System.out.println("DATASET " + datasetName + " FAILED");
            }
        }

        System.out.println(features);
        return features;
    }

    private static Instances initFeatures() {
        ArrayList<Attribute> attInfo = new ArrayList<>();
        for(int index = 0; index < 9; index++) {
            Attribute attribute = new Attribute("dataset" + index);
            attInfo.add(attribute);
        }
        attInfo.add(new Attribute("target"));
        Instances features = new Instances("features", attInfo, 20);
        features.setClassIndex(9);
        return features;
    }

    private static double[] getDatasetFeature(String dataset, boolean isTrain) throws Exception {
        Instances[] trainTest = loadDataset(dataset);

        double maxAcc = 0.0;
        String befitCls = classifier[0];
        for(String classifierName: classifier) {
            try {
                double acc = getClassifierAcc(classifierName, dataset, trainTest);
                if(acc > maxAcc) {
                    maxAcc = acc;
                    befitCls = classifierName;
                }
            } catch (Exception e) {
                System.out.println(classifierName + " on " + dataset + "\t\t FAILED ");
            }
        }

        double befit = 0;
        for(int i = 0; i < classifier.length; i++) {
            if(classifier[i].equals(befitCls)) {
                befit = i;
            }
        }

        return computeFeature(trainTest[0], trainTest[1], isTrain ? befit : maxAcc);
    }

    public static Instances[] loadDataset(String dataset) throws IOException {
        // We'll be loading the ItalyPowerDemand dataset which is distributed with this codebase
        String basePath = "dataset/";
        Instances train;
        Instances test;

        train = DatasetLoading.loadDataThrowable(basePath + dataset + "/" + dataset + "_TRAIN.arff");
        test = DatasetLoading.loadDataThrowable(basePath + dataset + "/" + dataset + "_TEST.arff");

        //resample
        return InstanceTools.resampleTrainAndTestInstances(train, test, seed);
    }

    public static double getClassifierAcc(String classifierName, String datasetName, Instances[] trainTest) throws Exception {
        Instances train = trainTest[0];
        Instances test = trainTest[1];

        Classifier classifier = ClassifierLists.setClassifierClassic(classifierName, seed);
        classifier.buildClassifier(train);

        // Setup the evaluator
        Evaluator testSetEval = new SingleTestSetEvaluator(seed, true, true);

        // And, in this case, test on the single held-out test set.
        ClassifierResults testResults = testSetEval.evaluate(classifier, test);
        System.out.println(classifierName + " accuracy on " + datasetName + ":\t\t" + testResults.getAcc());
        return testResults.getAcc();
    }

    //特征提取，两次使用随机投影
    private static double[] computeFeature(Instances train, Instances test, double befitCls) throws Exception {
        train.addAll(test);

        RandomProjection trans1 = new RandomProjection();
        trans1.setSeed(train.numAttributes());
        trans1.setInputFormat(train);
        trans1.setNumberOfAttributes(1);

        ArrayList<Attribute> attInfo = new ArrayList<>();
        for(int index = 0; index < train.numInstances(); index++) {
            attInfo.add(new Attribute("transInstance" + index));
        }
        Instances transResType = new Instances("transResType", attInfo, 1);

        double[] transResArray = new double[train.numInstances()];
        for(int index = 0; index < train.numInstances(); index++) {
            trans1.input(train.instance(index));
            Instance transInstance = trans1.output();
            transResArray[index] = transInstance.value(0);
        }
        Instance transRes = new DenseInstance(1.0, transResArray);
        transRes.setDataset(transResType);
        transResType.add(transRes);

        RandomProjection trans2 = new RandomProjection();
        trans2.setSeed(train.numInstances());
        trans2.setInputFormat(transResType);
        trans2.setNumberOfAttributes(8);

        double[] res = new double[10];
        trans2.input(transResType.instance(0));
        Instance transInstance = trans2.output();
        for(int index = 0; index < 8; index++) {
            res[index] = transInstance.value(index);
        }
        res[8] = train.numAttributes() * train.numInstances();
        res[9] = befitCls;

        return res;
    }
}
