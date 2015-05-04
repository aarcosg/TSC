package us.idinfor.tsclassification;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;

public class MainGTSRB {
	
	public static final String DIR_FEAT_TRAIN = "C:/Users/Alvaro/Downloads/GTSRB_Final_Training_HOG/GTSRB/Final_Training/HOG/HOG_03";
	public static final String DIR_FEAT_TEST = "C:/Users/Alvaro/Downloads/GTSRB_Final_Test_HOG/GTSRB/Final_Test/HOG/HOG_03";
	private static int nExample = 0;
	private static int nTestExample = 0;
	private static int nClasses = 0;
	private static int dim = 0;
	private static Mat trainingDataMat;
	private static Mat labelsMat;
	private static Mat testDataMat;

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		countTrainExample(DIR_FEAT_TRAIN);
		getFeatureVectorDimensionality(DIR_FEAT_TRAIN);
		loadTrainArrays(DIR_FEAT_TRAIN);
		System.out.println("Feature Vector Dimensionality = " + dim);
		System.out.println("Number Classes = " + nClasses);
		System.out.println("Number Training examples = " + nExample);
		
	    // Set up SVM's parameters
		CvSVMParams params = new CvSVMParams();
	    params.set_svm_type(CvSVM.C_SVC);
	    params.set_kernel_type(CvSVM.LINEAR);
	    TermCriteria criteria = new TermCriteria(TermCriteria.EPS, 100, 1e-6);
	    params.set_term_crit(criteria);

	    // Train the SVM
	    CvSVM SVM = new CvSVM();
	    System.out.println("Start SVM training...");
	    SVM.train(trainingDataMat, labelsMat, new Mat(), new Mat(), params);
	    System.out.println("SVM training finished");
	    SVM.save("svmhog3.xml");
	    
	    countTestExample(DIR_FEAT_TEST);
	    loadTestArrays(DIR_FEAT_TEST);
	    
	    Mat results = new Mat();
	    SVM.predict_all(testDataMat, results);
	    System.out.println("Prediction finished");
	    System.out.println("Writting output txt");
	    File fout = new File("outhog3.txt");
	    FileOutputStream fos = new FileOutputStream(fout);
	    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
	    for(int i = 0 ; i < results.rows() ; i++){
	    	String formattedNumber = String.format("%05d", i);
	    	bw.write(formattedNumber+".ppm; "+(int)results.get(i, 0)[0]);
	    	bw.newLine();
	    }
	    bw.close();
	    System.out.println("Writting finished");

	}
	
	private static void countTrainExample(String directory){
		for (int c = 0; ; c++){
			boolean foundFileForClass = false;
			for (int t = 0; ; t++){
				boolean foundFileForTrack = false;
				for (int e = 0; ; e++){
					String fileName = String.format("%s/%05d/%05d_%05d.txt", directory, c, t, e);
					if (new File(fileName).exists()){
						nExample++;
						foundFileForClass = true;
						foundFileForTrack = true;
					}
					else{
						break;
					}
				}
				if ( false == foundFileForTrack ) break;
			}
			if ( false == foundFileForClass ) break;
			nClasses++;
		}
	}
	
	private static void getFeatureVectorDimensionality(String directory) throws IOException{
		String fileName = String.format("%s/%05d/%05d_%05d.txt", directory, 0, 0, 0);
		File file = new File(fileName);
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

		if ( nExample == 39209 )
		{
			System.out.println("39209 training examples found. Everything looks fine.");
		}
		else if ( nExample == 0 )
		{
			System.out.println(" No training example found.");
		}
		else
		{
			System.out.println("Number of training examples read " + nExample +" is not equal 39209.");
		}
		
		trainingDataMat = new Mat(nExample, dim, CvType.CV_32FC1);
		labelsMat = new Mat(nExample,1,CvType.CV_32FC1);
		//testDataMat = new Mat(1,dim,CvType.CV_32FC1);
		
		int actualExample = 0;

		for (int c = 0; c < nClasses ; ++c)
		{
			for (int t = 0; ; ++t)
			{
				boolean foundFileForTrack = false;

				for (int e = 0; ; ++e)
				{
					String fileName = String.format("%s/%05d/%05d_%05d.txt", directory, c, t, e);
					File file = new File(fileName);
					int actualDim = 0;
					if(file.exists()){
						BufferedReader br = new BufferedReader(new FileReader(file));
						try {
						    for(String line; (line = br.readLine()) != null; ) {
						    	trainingDataMat.put(actualExample, actualDim, Double.valueOf(line));
						    	/*if(c == 2 && e == 5){
						    		testDataMat.put(0, actualDim, Double.valueOf(line));
						    	}*/
						    	++actualDim;
						    }
						    labelsMat.put(actualExample, 0, c);
						    ++actualExample;
						    foundFileForTrack = true;
						} finally {
							br.close();
						}
					} else {
						break;
					}
				}
				if ( false == foundFileForTrack ) break;
			}
		}
		System.out.println("Training arrays loaded");
		
	}
	
	private static void countTestExample(String directory){
		for (int e = 0; ; e++){
			String fileName = String.format("%s/%05d.txt", directory, e);
			if (new File(fileName).exists()){
				nTestExample++;
			}
			else{
				break;
			}
		}
	}
	
	private static void loadTestArrays(String directory) throws IOException
	{
		System.out.println("Loading testing arrays...");
		
		testDataMat = new Mat(nTestExample,dim,CvType.CV_32FC1);
		
		int actualExample = 0;
		for (int e = 0; ; ++e)
		{
			String fileName = String.format("%s/%05d.txt", directory, e);
			File file = new File(fileName);
			int actualDim = 0;
			if(file.exists()){
				BufferedReader br = new BufferedReader(new FileReader(file));
				try {
				    for(String line; (line = br.readLine()) != null; ) {
				    	testDataMat.put(actualExample, actualDim, Double.valueOf(line));
				    	++actualDim;
				    }
				    ++actualExample;
				} finally {
					br.close();
				}
			} else {
				break;
			}
		}

		System.out.println("Testing arrays loaded");
		
	}

}
