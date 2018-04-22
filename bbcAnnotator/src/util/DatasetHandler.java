package util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import javafx.util.Pair;


public class DatasetHandler {
	
	private ArrayList<Sample> samples;
	
	private Connection dbConnnection = null;
	private Statement dbStmt = null;
	
	public DatasetHandler() {
		
		// at first, wee need an array that contains all the paths for each file in the dataset
		ArrayList<Pair<Path, Boolean>> paths = new ArrayList<>();

		final File jarFile = new File(getClass().getProtectionDomain().getCodeSource().getLocation().getPath());
		final String testFolderName = "data/data_test";
		final String trainFolderName = "data/data_train";
		
		if(jarFile.isFile()) {  // Run with JAR file
		    boolean fsCreated = false;
		    final Map<String, String> env = new HashMap<>();
		    FileSystem fs = null;
			
			JarFile jar = null;
			try {
				jar = new JarFile(jarFile);
			
			    final Enumeration<JarEntry> entries = jar.entries(); //gives ALL entries in jar
			    while(entries.hasMoreElements()) {
			        final String name = entries.nextElement().getName();
			        
			        if (!name.startsWith("data"))
			        	continue;
			        
			        if (name.equals(testFolderName + "/") || name.equals(trainFolderName + "/")) // if isFolder
			        	continue;
	
        			String[] array;
					try {
						array = this.getClass().getClassLoader().getResource(name).toURI().toString().split("!");
						// dirty hack to deal with the filenames that contain '!'
						// http://bugs.java.com/view_bug.do?bug_id=4523159
						for (int i = 2; i < array.length; i++)
							array[1] += "!" + array[i];

						if (!fsCreated) {
							fs = FileSystems.newFileSystem(URI.create(array[0]), env);
							fsCreated = true;
						}
	        			Path path = fs.getPath(array[1]);
	        			if (name.startsWith(testFolderName + "/")) { //filter according to the path
			        		paths.add(new Pair<Path, Boolean>(path,false)); 
				        }
				        if (name.startsWith(trainFolderName + "/")) {
				        	paths.add(new Pair<Path, Boolean>(path,true));
				        }
					} catch (URISyntaxException | IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					};		
			    }

				jar.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else { // program run in IDE
			File testFolder = new File(getClass().getClassLoader().getResource("data/data_test").getFile());
			File trainFolder = new File(getClass().getClassLoader().getResource("data/data_train").getFile());
			
			for(File f : testFolder.listFiles()) {
        		paths.add(new Pair<Path, Boolean>(f.toPath(), false)); 
			}
			for(File f : trainFolder.listFiles()) {
        		paths.add(new Pair<Path, Boolean>(f.toPath(), false)); 
			}
		}
		
		// make sure we always have the same order
		Collections.sort(paths, Comparator.comparing(p -> p.getKey()));

		HashMap<String, Sample> samples = new HashMap<>();
		
		for (Pair<Path, Boolean> p : paths) {
			String[] filename = p.getKey().getFileName().toString().split("\\.(?=[^\\.]+$)");
			assert(filename.length == 2);
			String name = filename[0];
			String ext = filename[1];
			
			Sample currSample;
			if (samples.containsKey(name)) { 
				currSample = samples.get(name);
			} else {
				currSample = new Sample();
				currSample.name = name;
				currSample.isTrain = p.getValue();
				samples.put(name, currSample);
			}

			if (ext.equals("jpg")) {
				currSample.image = p.getKey();
			} else if (ext.equals("html")) {
				currSample.text = p.getKey();
			} else if (ext.equals("txt")) {
				currSample.caption = p.getKey();
			}
		}
		
		this.samples = new ArrayList<Sample>(samples.values());
		
		this.startAndReadDatabase();
	}
	
	public int getNumSamples() {
		return this.samples.size();
	}
	
	public Sample getSample(int index) {
		if (index < 0 || index >= this.samples.size())
			return null;
		
		Sample sample = this.samples.get(index);
		
		try {
			// check if sample was annotated before
			String sql = "SELECT * FROM Annotations WHERE name = '"+ sample.name +"'";
			ResultSet rs = this.dbStmt.executeQuery(sql);
						
			while (rs.next()) {
				String name = rs.getString(1);
				assert(name.equals(sample.name));
				int annotation = Integer.parseInt(rs.getString(2));
				sample.annotationMI = annotation;
				annotation = Integer.parseInt(rs.getString(3));
				sample.annotationSC = annotation;
				sample.imageType = rs.getString(4);
				//System.out.println(name + ", " + annotation);
			}
			
			// get text snippets
			sql = "SELECT * FROM Snippets WHERE name = '"+ sample.name +"'";
			rs = this.dbStmt.executeQuery(sql);
			
			sample.markedText.clear();
						
			while (rs.next()) {
				String snippet = rs.getString(3);
				snippet = snippet.replaceAll("''", "'");
				sample.markedText.add(snippet);
			}
			
			rs.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		
		return sample;
	}
	
	public void changeAnnotationMI(Sample sample, int annoMI) {
		if (sample.annotationMI == annoMI)
			return;
		
		sample.annotationMI = annoMI;
		
		this.backupSample(sample);
	}
	
	public void changeAnnotationSC(Sample sample, int annoSC) {
		if (sample.annotationSC == annoSC)
			return;
		
		sample.annotationSC = annoSC;
		
		this.backupSample(sample);
	}
	
	public void backupSample(Sample sample) {
		try {
			String sql = "SELECT * FROM Annotations WHERE name = '"+ sample.name +"'";
			ResultSet rs = this.dbStmt.executeQuery(sql);
			int i;
			if (!rs.next()) { // if no entry is in the dataset yet
		        sql = "INSERT INTO Annotations (name, annoMI, annoSC, type) VALUES ('"+ sample.name +"', "+ sample.annotationMI +", "+ sample.annotationSC +", '"+ sample.imageType +"')";
		        i = this.dbStmt.executeUpdate(sql);
		        if (i == -1) {
		            System.out.println("db error : " + sql);
		        }
			} else {
				int count = 0;
				do {
					count++;
				} while (rs.next());
				assert(count == 1);
				
				sql ="UPDATE Annotations SET annoMI = "+ sample.annotationMI + ", annoSC = "+ sample.annotationSC + ", type = '"+ sample.imageType + "' WHERE name = '"+ sample.name +"'";
				i = this.dbStmt.executeUpdate(sql);
		        if (i == -1) {
		            System.out.println("db error : " + sql);
		        }
			}
			
			// backup text snippets
			// remove existing ones
			sql ="DELETE FROM Snippets WHERE name = '"+ sample.name +"'";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
	        
	        // add current ones 
	        for (int j = 0; j < sample.markedText.size(); j++) {
	        	String snippet = sample.markedText.get(j);
	        	snippet = snippet.replaceAll("'", "''");
	        	sql = "INSERT INTO Snippets (name, index, snippet) VALUES ('"+ sample.name +"', "+ j +", '"+ snippet +"')";
		        i = this.dbStmt.executeUpdate(sql);
		        if (i == -1) {
		            System.out.println("db error : " + sql);
		        }
	        }
			
			rs.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}
	
	public void saveAnnotations(File file) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(file);
        StringBuilder sb = new StringBuilder();
        
        try {
	        for (int i = 0; i < this.samples.size(); i++) {
	        	Sample s = this.getSample(i);
	        	if (s.annotationMI != -1 || s.annotationSC != -3) {
	        		JSONObject jsonObj = new JSONObject();
	        		jsonObj.put("name", s.name);
	        		jsonObj.put("mi", s.annotationMI/4.0);
	        		jsonObj.put("sc", s.annotationSC/2.0);
	        		jsonObj.put("train", s.isTrain);
	        		jsonObj.put("type", s.imageType);
	        		
	        		JSONArray snippets = new JSONArray();
	        		
	                for (String m : s.markedText) {	                	
	                	snippets.put(m);
	                	if (m.contains("\n"))
	                		System.out.println(snippets);
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
        sb.append("// column anno_mi: Annotation for Mutual Information\n");
        sb.append("// column anno_sc: Annotation for Semantic Correlation\n"); 
        sb.append("// column isTrainingFile: Marks the folder, where the file came from\n"); 
        sb.append("// column type: Type of image\n");
        sb.append("// column markedSnippets: A list of strings from the document that were marked to be particularly high correlated to the image\n");
        
        sb.append("name");
        sb.append(',');
        sb.append("anno_mi");
        sb.append(',');
        sb.append("anno_sc");
        sb.append(',');
        sb.append("isTrainingFile");
        sb.append(',');
        sb.append("type");
        sb.append(',');
        sb.append("markedSnippets");
        sb.append('\n');
        
        for (int i = 0; i < this.samples.size(); i++) {
        	Sample s = this.getSample(i);
        	if (s.annotationMI != -1 || s.annotationSC != -3) {
        		sb.append(s.name);
                sb.append(',');
                sb.append(s.annotationMI);
                sb.append(',');
                sb.append(s.annotationSC/2.0);
                sb.append(',');
                sb.append(s.isTrain ? "1" : "0");
                sb.append(',');
                sb.append(s.imageType);
                for (String m : s.markedText) {
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

	public static String readCaption(Path captionFile) {
		byte[] encoded;
		try {
			encoded = Files.readAllBytes(captionFile);
			return new String(encoded, StandardCharsets.UTF_8);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return "";		
	}
	
	public static String readText(Path textFile) {
		byte[] encoded;
		try {
			encoded = Files.readAllBytes(textFile);
			return new String(encoded, StandardCharsets.ISO_8859_1);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return "";		
	}
	
	public void startAndReadDatabase() {

		try {
			// Treiberklasse laden
			Class.forName("org.hsqldb.jdbcDriver");
		} catch (ClassNotFoundException e) {
			System.err.println("Treiberklasse nicht gefunden!");
			return;
		}

		try {
			String settingsPath = CrossPlatform.pathToSettings();
			System.out.println(settingsPath);
			this.dbConnnection = DriverManager.getConnection("jdbc:hsqldb:file:"+settingsPath+"hsql; shutdown=true");
			this.dbStmt = this.dbConnnection.createStatement();
			String sql = "";
			int i;
			
//			sql = "DROP TABLE Annotations";
//			i = this.dbStmt.executeUpdate(sql);
//	        if (i == -1) {
//	            System.out.println("db error : " + sql);
//	        }
//			
//			sql = "DROP TABLE Snippets";
//			i = this.dbStmt.executeUpdate(sql);
//	        if (i == -1) {
//	            System.out.println("db error : " + sql);
//	        }

			// create table if not existing yeet
			sql = "CREATE TABLE IF NOT EXISTS Annotations (name varchar(1000) PRIMARY KEY, annoMI int, annoSC int, type varchar(1000))";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
	        
	        sql = "CREATE TABLE IF NOT EXISTS Snippets (name varchar(1000), index int, snippet varchar(10000))";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
			
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}
	
	public void closeDB() {
		if (this.dbConnnection != null) {
			try {
				this.dbConnnection.close();
			} catch (SQLException e) {
				e.printStackTrace();
			}
		}
		if (this.dbStmt != null) {
			try {
				this.dbStmt.close();
			} catch (SQLException e) {
				e.printStackTrace();
			}
		}
	}
}

