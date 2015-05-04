package us.idinfor.tsclassification;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

public class HogExtractor {

	public static void main(String[] args) throws IOException {
		//Load OpenCV core library
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		//Output directory which will contain extracted features files.
		File dirOut = new File("C:/Users/Alvaro/Desktop/Spain/Testing/HOG");
		//Input images directory.
		File dir = new File("C:/Users/Alvaro/Desktop/Spain/Testing/Images");
		/*
		 * Setup HOGDescriptor to calculate HOG features.
		 * That config will generate 2916-dimensional feature vectors.
		 * Same dimensionality is used in: 
		 * J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. 
		 * The German Traffic Sign Recognition Benchmark: A multi-class classification competition.
		 * In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011.
		 * */
		HOGDescriptor hog = new HOGDescriptor(new Size(40,40), new Size(8,8), new Size(4,4), new Size(4,4), 9);
		//Iterate over class directories
		if(dir.isDirectory()){
			for(File classDir : dir.listFiles()){
				if(classDir.isDirectory()){
					System.out.println("Processing dir: " + classDir.getName());
					//Create new directory per each class
					File hogDir = new File(dirOut.getAbsolutePath()+File.separator+classDir.getName());
					hogDir.mkdir();
					//Iterate over images
					for(File image : classDir.listFiles()){
						if(image.isFile() && (image.getName().endsWith(".ppm") || image.getName().endsWith(".jpg"))){
							   //Load grayscaled image
							   Mat img = Highgui.imread(image.getAbsolutePath(), Highgui.CV_LOAD_IMAGE_GRAYSCALE);
							   //Resize image to 40x40 (see referenced paper)
							   Imgproc.resize(img, img, new Size(40,40));
							   MatOfFloat descriptors = new MatOfFloat();
							   //Calculate HOG features of the image
							   hog.compute(img, descriptors);
							   //Create new file where features will be stored
							   File fDescriptors = new File(hogDir.getAbsolutePath()+File.separator+image.getName().substring(0, image.getName().indexOf("."))+".txt");
							   FileOutputStream fos = new FileOutputStream(fDescriptors);
							   BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
							   int i = 0;
							   //Iterate over features and write them to the file (one per line)
							   for(Float feature : descriptors.toList()){
								   bw.write(feature.toString());
								   if(i<hog.getDescriptorSize()-1){
									   bw.newLine();
								   }
								   i++;
							   }
							   bw.close();
						}
					}
				}
			}
		}

	}

}
