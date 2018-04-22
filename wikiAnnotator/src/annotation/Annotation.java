package annotation;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;

/*
 * Image annotation
 */
public class Annotation {
	// article id
	public int id = -1;
	// section heading (stays null for article images
	public String section = null;
	// image name (filename)
	public String name = null;
	
	// mutual information
	public int annotationMI = -1; // -1 is unknown
	// semantic correlation
	public int annotationSC = -3; // -3 is unknown
	// text, that was marked to have a particular high correlation with the image
	//public ArrayList<String> markedText = new ArrayList<String>();
	public ObservableList<String> markedText = FXCollections.observableArrayList();
	
	public boolean validSample = true;
	
	public String imageType = "Photograph";
}
