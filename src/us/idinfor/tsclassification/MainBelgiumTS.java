package us.idinfor.tsclassification;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;

public class MainBelgiumTS {
	
	//Training directory
	public static final String DIR_FEAT_TRAIN = "D:/BelgiumTS Dataset/TSImages/Classification/HOG/Training";
	//Testing directory
	public static final String DIR_FEAT_TEST = "D:/BelgiumTS Dataset/TSImages/Classification/HOG/Testing";
	//Total number of training images
	private static int nExample = 0;
	//Total number of testing images
	private static int nTestExample = 0;
	//Total number of classes
	private static int nClasses = 0;
	//Dimensionality of features vector
	private static int dim = 0;
	//Training features vectors Mat
	private static Mat trainingDataMat;
	//Class labels Mat (eg: 0,1,2,3...)
	private static Mat labelsMat;
	//Testing features vectors Mat
	private static Mat testDataMat;
	//Map used to keep a reference of filenames 
	private static Map<Integer,String> testFiles = new HashMap<Integer, String>();

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		//Optional function to create a unique .csv annotations file (annotations in belgium dataset are split up in different directories) 
		createGroundTruth("D:/BelgiumTS Dataset/TSImages/Classification/Testing");
		
		//Count training examples
		countTrainExample(DIR_FEAT_TRAIN);
		//Calculate features vector dimensionality
		getFeatureVectorDimensionality(DIR_FEAT_TRAIN);
		//Load training features vectors (stored in txt files) into arrays
		loadTrainArrays(DIR_FEAT_TRAIN);
		System.out.println("Feature Vector Dimensionality = " + dim);
		System.out.println("Number Classes = " + nClasses);
		System.out.println("Number Training examples = " + nExample);
		
	    //Set up SVM's parameters
		CvSVMParams params = new CvSVMParams();
	    params.set_svm_type(CvSVM.C_SVC);
	    params.set_kernel_type(CvSVM.LINEAR);
	    TermCriteria criteria = new TermCriteria(TermCriteria.EPS, 100, 1e-6);
	    params.set_term_crit(criteria);

	    //Train the SVM
	    CvSVM SVM = new CvSVM();
	    System.out.println("Start SVM training...");
	    SVM.train(trainingDataMat, labelsMat, new Mat(), new Mat(), params);
	    System.out.println("SVM training finished");
	    //Save trained model so we will not need to train it again, just load it
	    SVM.save("belgiumsvmhog.xml");
	    
	    //Count testing examples
	    countTestExample(DIR_FEAT_TEST);
	    //Load testing features vectors (stored in txt files) into arrays
	    loadTestArrays(DIR_FEAT_TEST);
	    
	    Mat results = new Mat();
	    //Per each features vector (images), predict its class
	    SVM.predict_all(testDataMat, results);
	    System.out.println("Prediction finished");
	    System.out.println("Writting output txt");
	    //Create results txt file
	    File fout = new File("outbelgiumhog.txt");
	    FileOutputStream fos = new FileOutputStream(fout);
	    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
	    for(int i = 0 ; i < results.rows() ; i++){
	    	bw.write(testFiles.get(i)+";"+(int)results.get(i, 0)[0]);
	    	bw.newLine();
	    }
	    bw.close();
	    System.out.println("Writting finished");

	}
	
	private static void countTrainExample(String directory){
		File dir = new File(directory);
		if(dir.isDirectory()){
			for(File classDir : dir.listFiles()){
				if(classDir.isDirectory()){
					nExample += classDir.listFiles().length;
				}
				nClasses++;
			}
		}
	}
	
	private static void getFeatureVectorDimensionality(String directory) throws IOException{
		File dir = new File(directory);
		File file = dir.listFiles()[0].listFiles()[0];
		BufferedReader br = new BufferedReader(new FileReader(file));
		try {
		    while(br.readLine() != null) {
		    	++dim;
		    }
		} finally {
			br.close();
		}
	}
	
	private static void loadTrainArrays(String directory) throws IOException
	{
		System.out.println("Loading training arrays...");

		if ( nExample == 4575 )
		{
			System.out.println("4575 training examples found. Everything looks fine.");
		}
		else if ( nExample == 0 )
		{
			System.out.println(" No training example found.");
		}
		else
		{
			System.out.println("Number of training examples read " + nExample +" is not equal 4575.");
		}
		
		trainingDataMat = new Mat(nExample, dim, CvType.CV_32FC1);
		labelsMat = new Mat(nExample,1,CvType.CV_32FC1);
		
		File dir = new File(directory);
		int actualExample = 0;
		if(dir.isDirectory()){
			for(File classDir : dir.listFiles()){
				if(classDir.isDirectory()){
					for(File features : classDir.listFiles()){
						if(features.isFile() && features.getName().endsWith(".txt")){
							int actualDim = 0;
							BufferedReader br = new BufferedReader(new FileReader(features));
							try {
							    for(String line; (line = br.readLine()) != null; ) {
							    	trainingDataMat.put(actualExample, actualDim, Double.valueOf(line));
							    	++actualDim;
							    }
							    labelsMat.put(actualExample, 0, Integer.valueOf(classDir.getName()));
							    ++actualExample;
							} finally {
								br.close();
							}
						}
					}
				}
			}
		}
		
		System.out.println("Training arrays loaded");
		
	}
	
	private static void countTestExample(String directory){
		File dir = new File(directory);
		if(dir.isDirectory()){
			for(File classDir : dir.listFiles()){
				if(classDir.isDirectory()){
					nTestExample += classDir.listFiles().length;
				}
			}
		}
	}
	
	private static void loadTestArrays(String directory) throws IOException
	{
		System.out.println("Loading testing arrays...");
		
		testDataMat = new Mat(nTestExample,dim,CvType.CV_32FC1);
		
		File dir = new File(directory);
		int actualExample = 0;
		if(dir.isDirectory()){
			for(File classDir : dir.listFiles()){
				if(classDir.isDirectory()){
					for(File features : classDir.listFiles()){
						if(features.isFile() && features.getName().endsWith(".txt")){
							int actualDim = 0;
							BufferedReader br = new BufferedReader(new FileReader(features));
							try {
							    for(String line; (line = br.readLine()) != null; ) {
									testFiles.put(actualExample,classDir.getName()+"/"+features.getName().replace(".txt", ".ppm"));
							    	testDataMat.put(actualExample, actualDim, Double.valueOf(line));
							    	++actualDim;
							    }
							    ++actualExample;
							} finally {
								br.close();
							}
						}
					}
				}
			}
		}

		System.out.println("Testing arrays loaded");
		
	}
	
	private static void createGroundTruth(String directory) throws IOException{
	    File fout = new File("belgiumtestingGT.csv");
	    FileOutputStream fos = new FileOutputStream(fout);
	    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
	    
		File dir = new File(directory);
		if(dir.isDirectory()){
			for(File classDir : dir.listFiles()){
				if(classDir.isDirectory()){
					for(File file : classDir.listFiles()){
						if(file.isFile() && file.getName().endsWith(".csv")){
							BufferedReader br = new BufferedReader(new FileReader(file));
							try {
							    for(String line; (line = br.readLine()) != null; ) {
							    	if(!line.startsWith("Filename")){
								    	bw.write(classDir.getName()+"/"+line);
								    	bw.newLine();
							    	}
							    }
							} finally {
								br.close();
							}
						}
					}
				}
			}
		}
		bw.close();
	}
	

}
