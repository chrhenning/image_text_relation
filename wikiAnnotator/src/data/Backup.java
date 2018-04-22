package data;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import annotation.Annotation;
import util.CrossPlatform;

public class Backup {
	// singleton class
	private static Backup instance = null;
	
	private Connection dbConnnection = null;
	private Statement dbStmt = null;
	
	private Backup() {

		try {
			// load driver class
			Class.forName("org.hsqldb.jdbcDriver");
		} catch (ClassNotFoundException e) {
			System.err.println("Driver class not found!");
			return;
		}

		try {
			String settingsPath = CrossPlatform.pathToSettings();
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
//			
//			sql = "DROP TABLE Settings";
//			i = this.dbStmt.executeUpdate(sql);
//	        if (i == -1) {
//	            System.out.println("db error : " + sql);
//	        }

			// create table if not existing yet
			sql = "CREATE TABLE IF NOT EXISTS Annotations (id int, section varchar(1000), name varchar(1000), annoMI int, annoSC int, valid int, type varchar(1000))";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
	        
	        sql = "CREATE TABLE IF NOT EXISTS Snippets (id int, section varchar(1000), name varchar(1000), index int, snippet varchar(10000))";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
	        
	        sql = "CREATE TABLE IF NOT EXISTS Settings (name varchar(1000) PRIMARY KEY, setting varchar(10000))";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
			
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}
	
	public static Backup getInstance() {
		if (instance == null) {
			instance = new Backup();
		} 
		return instance;
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
	
	public void clearDB() {
		try {
			String sql = "DROP TABLE Annotations";
			int i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
			
			sql = "DROP TABLE Snippets";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
	        
	        sql = "CREATE TABLE IF NOT EXISTS Annotations (id int, section varchar(1000), name varchar(1000), annoMI int, annoSC int, valid int, type varchar(1000))";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
	        
	        sql = "CREATE TABLE IF NOT EXISTS Snippets (id int, section varchar(1000), name varchar(1000), index int, snippet varchar(10000))";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void setDataSetPath(String path) {
		try {			
			String sql = "SELECT * FROM Settings WHERE name = 'dsPath'";
			ResultSet rs = this.dbStmt.executeQuery(sql);
			int i;
			if (!rs.next()) { // if no entry is in the dataset yet
		        sql = "INSERT INTO Settings (name, setting) VALUES ('dsPath', '"+ path +"')";
		        i = this.dbStmt.executeUpdate(sql);
		        if (i == -1) {
		            System.out.println("db error : " + sql);
		        }
			} else {				
				sql ="UPDATE Settings SET setting = '"+ path + "' WHERE name = 'dsPath'";
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
	
	public String getDataSetPath() {
		String ret = null;
		
		try {
			// check if sample was annotated before
			String sql = "SELECT * FROM Settings WHERE name = 'dsPath'";
			ResultSet rs = this.dbStmt.executeQuery(sql);
						
			while (rs.next()) 
				ret = rs.getString(2);
		
			rs.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		
		return ret;
	}
	
	public Annotation getAnnotation(int id, String section, String name) {

		Annotation anno = new Annotation();
		anno.id = id;
		anno.section = section;
		anno.name = name;
		
		try {
			// check if sample was annotated before
			String sql = "SELECT * FROM Annotations WHERE id = "+ id +" AND section = '"+ section.replaceAll("'", "''") +"' AND name = '"+ name.replaceAll("'", "''") +"'";
			ResultSet rs = this.dbStmt.executeQuery(sql);

			while (rs.next()) {
				anno.annotationMI = Integer.parseInt(rs.getString(4));
				anno.annotationSC = Integer.parseInt(rs.getString(5));
				anno.validSample = Integer.parseInt(rs.getString(6)) > 0;
				anno.imageType = rs.getString(7);
			}
			
			// get text snippets
			sql = "SELECT * FROM Snippets WHERE id = "+ id +" AND section = '"+ section.replaceAll("'", "''") +"' AND name = '"+ name.replaceAll("'", "''") +"'";
			rs = this.dbStmt.executeQuery(sql);
			
			anno.markedText.clear();
						
			while (rs.next()) {
				String snippet = rs.getString(5);
				snippet = snippet.replaceAll("''", "'");
				anno.markedText.add(snippet);
			}
			
			rs.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		
		return anno;
	}
	
	public void backupAnnotation(Annotation anno) {
		try {
			String sql = "SELECT * FROM Annotations WHERE id = "+ anno.id +" AND section = '"+ anno.section.replaceAll("'", "''") +"' AND name = '"+ anno.name.replaceAll("'", "''") +"'";
			ResultSet rs = this.dbStmt.executeQuery(sql);
			int i;
			if (!rs.next()) { // if no entry is in the dataset yet
		        sql = "INSERT INTO Annotations (id, section, name, annoMI, annoSC, valid, type) VALUES ("+ anno.id +", '"+ anno.section.replaceAll("'", "''") +"', '"+ anno.name.replaceAll("'", "''") +"', "+ anno.annotationMI +", "+ anno.annotationSC +", "+ (anno.validSample?1:0) +", '"+ anno.imageType +"')";
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
				
				sql ="UPDATE Annotations SET annoMI = "+ anno.annotationMI + ", annoSC = "+ anno.annotationSC + ", valid = "+ (anno.validSample?1:0) + ", type = '"+ anno.imageType + "' WHERE id = "+ anno.id +" AND section = '"+ anno.section.replaceAll("'", "''") +"' AND name = '"+ anno.name.replaceAll("'", "''") +"'";
				i = this.dbStmt.executeUpdate(sql);
		        if (i == -1) {
		            System.out.println("db error : " + sql);
		        }
			}
			
			// backup text snippets
			// remove existing ones
			sql ="DELETE FROM Snippets WHERE id = "+ anno.id +" AND section = '"+ anno.section.replaceAll("'", "''") +"' AND name = '"+ anno.name.replaceAll("'", "''") +"'";
			i = this.dbStmt.executeUpdate(sql);
	        if (i == -1) {
	            System.out.println("db error : " + sql);
	        }
	        
	        // add current ones 
	        for (int j = 0; j < anno.markedText.size(); j++) {
	        	String snippet = anno.markedText.get(j);
	        	snippet = snippet.replaceAll("'", "''");
	        	sql = "INSERT INTO Snippets (id, section, name, index, snippet) VALUES ("+ anno.id +", '"+ anno.section.replaceAll("'", "''") +"', '"+ anno.name.replaceAll("'", "''") +"', "+ j +", '"+ snippet +"')";
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
	
	// get all Annotations
	ArrayList<Annotation> getAnnotations() {
		ArrayList<Annotation> annos = new ArrayList<>();
		
		try {
			// check if sample was annotated before
			String sql = "SELECT * FROM Annotations";
			ResultSet rs = this.dbStmt.executeQuery(sql);

			while (rs.next()) {
				Annotation anno = this.getAnnotation(Integer.parseInt(rs.getString(1)),rs.getString(2),rs.getString(3));
				annos.add(anno);
			}
			
			rs.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		
		return annos;
	}
}

