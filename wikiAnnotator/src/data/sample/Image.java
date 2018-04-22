package data.sample;

import java.nio.file.Path;
import java.util.ArrayList;

public class Image {
	public Image() {
		keyphrases = new ArrayList<String>(); 
	}
	
	public String caption;
	public String name;
	public Path imgfile;
	public Path metafile;
	public ArrayList<String> keyphrases;
	
}
