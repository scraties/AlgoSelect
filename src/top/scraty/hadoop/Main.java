package top.scraty.hadoop;

import experiments.data.DatasetLoading;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

public class Main {
    private final static String[] train = {"GunPoint", "Ham", "ToeSegmentation1", "UMD", "Wine",
                            "BirdChicken", "BME", "Car", "CBF", "Chinatown",
                            "Coffee", "DiatomSizeReduction", "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend"};
    private final static String[] test = {"OliveOil", "Beef", "BeetleFly", "Meat", "ArrowHead"};

    private static void writeToFile(String content, String fileName) throws FileNotFoundException {
        PrintWriter out = new PrintWriter("dataset/" + fileName);
        out.print(content);
        out.close();
    }

    private static void getTrain() {
        System.out.println("GET TRAIN");
        Instances features = DatasetFeature.getFeatures(train, true);
        if(features == null) return;
        try {
            writeToFile(features.toString(), "TRAIN.arff");
        } catch (FileNotFoundException e) {
            System.out.println("write to TRAIN.arff FAILED");
        }
    }

    private static void getTest() {
        System.out.println("GET TEST");
        Instances features = DatasetFeature.getFeatures(test, false);
        if(features == null) return;
        try {
            writeToFile(features.toString(), "TEST.arff");
        } catch (FileNotFoundException e) {
            System.out.println("write to TEST.arff FAILED");
        }
    }

    private static void getResult() {
        System.out.println("GET RESULT");
        Instances trainFeatures, testFeatures;
        try {
            trainFeatures = DatasetLoading.loadDataThrowable("dataset/" + "TRAIN.arff");
            testFeatures = DatasetLoading.loadDataThrowable("dataset/" + "TEST.arff");
        } catch (IOException e) {
            System.out.println("load TRAIN.arff and TEST.arff: FAILED");
            return;
        }

        RandomForest randomForest = new RandomForest();
        randomForest.setNumIterations(500);
        randomForest.setSeed(DatasetFeature.seed);

        try {
            randomForest.buildClassifier(trainFeatures);
        } catch (Exception e) {
            System.out.println("Train of RandomForest: FAILED");
        }

        try {
            AlgoEvaluator.evaluate(randomForest, test, testFeatures);
        } catch (Exception e) {
            System.out.println("Evaluate: FAILED");
        }
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        String cmd = in.nextLine();
        switch (cmd) {
            case "get train":
                getTrain();
                break;
            case "get test":
                getTest();
                break;
            case "get result":
                getResult();
                break;
            case "get all":
                getTrain();
                getTest();
                getResult();
                break;
        }
        in.close();
    }
}
