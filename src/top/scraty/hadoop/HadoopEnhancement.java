package top.scraty.hadoop;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class HadoopEnhancement {

    public static boolean runHadoop(boolean isTrain) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Time Series Classification");
        job.setJarByClass(HadoopEnhancement.class);
        job.setMapperClass(HadoopEnhancement.AlgoMapper.class);
        job.setReducerClass(HadoopEnhancement.AlgoReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path("hadoop_in/in.txt"));
        FileOutputFormat.setOutputPath(job, new Path("hadoop_out_" + (isTrain ? "train":"test")));
        return job.waitForCompletion(true);
    }

    public static class AlgoReducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context) throws IOException, InterruptedException {
            double maxAcc = 0.0, befitNum = 0.0;
            String befit = "";

            for(Text value : values) {
                String[] now = value.toString().split(" ");
                double acc = Double.parseDouble(now[1]);
                if (acc > maxAcc) {
                    befit = now[0];
                    maxAcc = acc;
                }
            }

            for(int i = 0; i < DatasetFeature.classifier.length; i++) {
                if(DatasetFeature.classifier[i].equals(befit)) {
                    befitNum = i;
                }
            }

            context.write(key, new Text(befitNum + "\t" + maxAcc));
        }
    }

    public static class AlgoMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Mapper<LongWritable, Text, Text, Text>.Context context) throws IOException, InterruptedException {
            double acc;
            String[] values = value.toString().split(" ");
            try {
                acc = DatasetFeature.getClassifierAcc(values[1], values[0], DatasetFeature.loadDataset(values[0]));
            } catch (Exception e) {
                System.out.println(values[1] + " on " + values[0] + "\t\t FAILED ");
                acc = 0.0;
            }

            context.write(new Text(values[0]), new Text(values[1] + " " + acc));
        }
    }
}

