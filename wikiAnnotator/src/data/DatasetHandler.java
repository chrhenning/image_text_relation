package data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.logging.Logger;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import annotation.Annotation;
import data.sample.Image;
import data.sample.Sample;
import data.sample.Section;


public class DatasetHandler {
	
	// singleton
	private static DatasetHandler instance = null;
	
	private File dsDir; // directory of data set	
	private ArrayList<Sample> samples;
	
	private final static Logger LOGGER = Logger.getLogger(DatasetHandler.class.getName());
	
	private DatasetHandler() {
		
	}
	
	public static DatasetHandler getInstance() {
		if (instance == null)
			instance = new DatasetHandler();
		return instance;
	}
	
	public void loadDataset(File jsonFile) {
		dsDir = jsonFile.getParentFile();
		Backup.getInstance().setDataSetPath(jsonFile.toString());
		
		LOGGER.info("Reading dataset from: " + dsDir.toString());
		
		this.samples = new ArrayList<>();
		
		try (BufferedReader br = new BufferedReader(new FileReader(jsonFile))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		    	Sample sample = new Sample();
		    	JSONObject jobj = new JSONObject(line);
		    	sample.id = jobj.getInt("id");
		    	sample.title = jobj.getString("title");
		    	JSONArray sec = jobj.getJSONArray("sections");
				for (int i = 0; i < sec.length(); i++) 
					sample.sections.add(this.toSection(sec.getJSONObject(i)));
				JSONArray ref = jobj.getJSONArray("references");
				for (int i = 0; i < ref.length(); i++) 
					sample.references.add(ref.getString(i));
				sample.summary = jobj.getString("summary");
				JSONArray keyphrases = jobj.getJSONArray("keyphrases");
				for (int i = 0; i < keyphrases.length(); i++) 
					sample.keyphrases.add(keyphrases.getString(i));
				sample.summary = jobj.getString("summary");
				sample.url = jobj.getString("url");
				JSONArray images = jobj.getJSONArray("images");
				for (int i = 0; i < images.length(); i++) 
					sample.images.add(this.toImage(images.getJSONObject(i)));
				try {
					JSONArray categories = jobj.getJSONArray("categories");
					for (int i = 0; i < categories.length(); i++) 
						sample.categories.add(categories.getString(i));
				} catch (JSONException e) {
					// ignore exception, since we where missing to download the categories for most of the dataset
				}
				
				this.samples.add(sample);
		    }
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public Section toSection(JSONObject jsec) throws JSONException {
		Section sec = new Section();
		
		sec.title = jsec.getString("title");
		sec.text = jsec.getString("text");
		JSONArray subsec = jsec.getJSONArray("subsections");
		for (int i = 0; i < subsec.length(); i++) 
			sec.subsections.add(this.toSection(subsec.getJSONObject(i)));
		JSONArray images = jsec.getJSONArray("images");
		for (int i = 0; i < images.length(); i++) 
			sec.images.add(this.toImage(images.getJSONObject(i)));
		JSONArray lists = jsec.getJSONArray("lists");
		for (int i = 0; i < lists.length(); i++)  {
			ArrayList<String> slist = new ArrayList<>();
			sec.lists.add(slist);
			JSONArray list = lists.getJSONArray(i);
			for (int j = 0; j < list.length(); j++)  {
				slist.add(list.getString(j));
			}
		}
		JSONArray keyphrases = jsec.getJSONArray("keyphrases");
		for (int i = 0; i < keyphrases.length(); i++) 
			sec.keyphrases.add(keyphrases.getString(i));
		
		return sec;
	}
	
	public Image toImage(JSONObject jimg) throws JSONException {
		Image img = new Image();
		
		img.caption = jimg.getString("caption");
		img.name = jimg.getString("filename");
		if (jimg.has("imgpath")) {
			assert(jimg.has("metapath"));
			img.imgfile = new File(this.dsDir, jimg.getString("imgpath")).toPath();
			img.metafile = new File(this.dsDir, jimg.getString("metapath")).toPath();
		} else {
			img.imgfile = null;
			img.metafile = null;
		}
		
		JSONArray keyphrases = jimg.getJSONArray("keyphrases");
		for (int i = 0; i < keyphrases.length(); i++) 
			img.keyphrases.add(keyphrases.getString(i));
		
		if (img.imgfile != null) {
			File imgFile = new File(img.imgfile.toString());
			File metaFile = new File(img.metafile.toString());
			
			if (!imgFile.exists() || !metaFile.exists()) {
				LOGGER.warning("Image file " + imgFile.toString() + " or its meta file does not exist.");
				img.imgfile = null;
				img.metafile = null;
			}
		}
		
		return img;
	}
	
	public Sample getSample(int index) {
		if (index < 0 || index >= this.samples.size())
			return null;
		
		Sample sample = this.samples.get(index);
		
		return sample;
	}
	
	public int getNumSamples() {
		return this.samples.size();
	}	
	
	public void saveAnnotations(File file) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(file);
        StringBuilder sb = new StringBuilder();
        
        try {
	        ArrayList<Annotation> annos = Backup.getInstance().getAnnotations();
	        for (Annotation a : annos) {
	        	if (a.annotationMI != -1 || a.annotationSC != -3 || !a.validSample) {
	        		JSONObject jsonObj = new JSONObject();
	        		jsonObj.put("id", a.id);
	        		jsonObj.put("section", a.section);
	        		jsonObj.put("name", a.name);
	        		jsonObj.put("mi", a.annotationMI/4.0);
	        		jsonObj.put("sc", a.annotationSC/2.0);
	        		jsonObj.put("valid", a.validSample);
	        		jsonObj.put("type", a.imageType);
	        		
	        		JSONArray snippets = new JSONArray();
	        		
	                for (String m : a.markedText) {	                	
	                	snippets.put(m);
	                }
	                
	                jsonObj.put("snippets",snippets);
	                
	                sb.append(jsonObj);
	                sb.append('\n');
	        	}
	        }
        } catch (JSONException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

        pw.write(sb.toString());
        pw.close();
	}	
	
	/*
	public void saveAnnotations(File file) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(file);
        StringBuilder sb = new StringBuilder();
        
        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        Calendar cal = Calendar.getInstance();
        
        sb.append("// bbcAnnotator; File created: " + dateFormat.format(cal.getTime()) + "\n");
        sb.append("// column id: Article id of sample\n");
        sb.append("// column section: Section title, where image is depicted ('Article', if image belonged to the whole article)\n");
        sb.append("// column name: Name of image\n");
        sb.append("// column anno_mi: Annotation for Mutual Information\n");
        sb.append("// column anno_sc: Annotation for Semantic Correlation\n"); 
        sb.append("// column valid: If sample is recommended for being used in classification tasks\n");
        sb.append("// column type: Type of image\n");
        sb.append("// column markedSnippets: A list of strings from the document that were marked to be particularly high correlated to the image\n");
        
        sb.append("id");
        sb.append(',');
        sb.append("section");
        sb.append(',');
        sb.append("name");
        sb.append(',');
        sb.append("anno_mi");
        sb.append(',');
        sb.append("anno_sc");
        sb.append(',');
        sb.append("valid");
        sb.append(',');
        sb.append("type");
        sb.append(',');
        sb.append("markedSnippets");
        sb.append('\n');
        
        ArrayList<Annotation> annos = Backup.getInstance().getAnnotations();
        for (Annotation a : annos) {
        	if (a.annotationMI != -1 || a.annotationSC != -3 || !a.validSample) {
        		sb.append(a.id);
                sb.append(',');
                sb.append(a.section);
                sb.append(',');
        		sb.append(a.name);
                sb.append(',');
                sb.append(a.annotationMI);
                sb.append(',');
                sb.append(a.annotationSC/2.0);
                sb.append(',');
                sb.append((a.validSample?1:0));
                sb.append(',');
                sb.append(a.imageType);
                for (String m : a.markedText) {
                	sb.append(',');
                	String escaped = m.replaceAll("(\\r|\\n|\\r\\n)+", "\\\\n");
                	escaped = escaped.replaceAll("\\\"", "\\\\\"");
                    sb.append("\""+escaped+"\"");
                }
                sb.append('\n');
        	}
        }

        pw.write(sb.toString());
        pw.close();
	}	
	*/
}

