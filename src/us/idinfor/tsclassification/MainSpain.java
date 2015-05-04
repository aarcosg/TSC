package us.idinfor.tsclassification;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.ml.CvSVM;

public class MainSpain {
	
	public static final String DIR_FEAT_TEST = "C:/Spain/Testing/HOG";
	private static int nTestExample = 0;
	private static Mat testDataMat;
	private static int dim = 0;
	private static Map<Integer,String> testFiles = new HashMap<Integer, String>();

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		setUpTestingImages();
		
		getFeatureVectorDimensionality(DIR_FEAT_TEST);
		System.out.println("Feature Vector Dimensionality = " + dim);

	    // Train the SVM
	    CvSVM SVM = new CvSVM();
	    System.out.println("Load germany SVM model");
	    SVM.load("svmhog3.xml");
	    
	    countTestExample(DIR_FEAT_TEST);
	    loadTestArrays(DIR_FEAT_TEST);
	    
	    Mat results = new Mat();
	    SVM.predict_all(testDataMat, results);
	    System.out.println("Prediction finished");
	    System.out.println("Writting output txt");
	    File fout = new File("outspain_germanysvm.txt");
	    FileOutputStream fos = new FileOutputStream(fout);
	    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
	    for(int i = 0 ; i < results.rows() ; i++){
	    	bw.write(testFiles.get(i)+";"+(int)results.get(i, 0)[0]);
	    	bw.newLine();
	    }
	    bw.close();
	    System.out.println("Writting finished");
		

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
									testFiles.put(actualExample,classDir.getName()+"/"+features.getName().replace(".txt", ".jpg"));
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
	
	private static void setUpTestingImages() throws IOException{
		File oldDir = new File ("C:/Imagenes señales España/Outputs");
		File newDir = new File ("C:/TestingSpain");
		int classes = 0;
		for(File dirClass : oldDir.listFiles()){
			if(dirClass.isDirectory()){
				for(File featureDir : dirClass.listFiles()){
					if(featureDir.isDirectory() && featureDir.getName().equalsIgnoreCase("smallBG")){
						for(File image : featureDir.listFiles()){
							if(image.getName().endsWith(".jpg")){
								String formattedNumber = String.format("%05d", classes);
								String pathString = newDir.getAbsolutePath() + File.separator + formattedNumber;
								File pathFile = new File(pathString);
								if(!pathFile.isDirectory()){
									pathFile.mkdir();
								}
								copyFile(image, new File(pathFile.getAbsolutePath() + File.separator + image.getName()));
							}
						}
					}
				}
			}
			classes++;
		}
	}
	
	private static void copyFile(File source, File dest) throws IOException {
	    InputStream is = null;
	    OutputStream os = null;
	    try {
	        is = new FileInputStream(source);
	        os = new FileOutputStream(dest);
	        byte[] buffer = new byte[1024];
	        int length;
	        while ((length = is.read(buffer)) > 0) {
	            os.write(buffer, 0, length);
	        }
	    } finally {
	        is.close();
	        os.close();
	    }
	}

}
