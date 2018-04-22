package util;

import java.io.File;

/**
 * Utilities to ensure platform independing handling of filenames.
 */
public class CrossPlatform {
    
    /**
     * This method returns the system dependent path to the settings, that this program has to store.
     * 
     * @return The directory, which contains the system settings. 
     * @author Christian Henning
     */
    public static String pathToSettings() {
    	// this variable should contain the path to the game settings
    	String settingsDirectory;
    	// get name of current operating system
    	String operatingSystem = (System.getProperty("os.name")).toUpperCase();
    	// find out, which settingsDirectory we need.
    	// This depends of the current OS.
    	if (operatingSystem.contains("WIN")) {
    		// current OS is windows
    	    // we store the settings into the AppData folder
    	    settingsDirectory = System.getenv("AppData");
    	    
    	    // we store the settings into this subfolder
        	settingsDirectory += "/bbcAnnotator/";
    	} else {
    		if (operatingSystem.contains("MAC")) {
	    		// current OS is Mac os
	    		// we are starting in the user's home directory
	    	    settingsDirectory = System.getProperty("user.home");
	    	    // we store the settings into this folder
	    	    settingsDirectory += "/Library/Application Support";
	    	} else {
	    		// in any other case (mostly linux)
	    	    // we use simply the home directory
	    	    settingsDirectory = System.getProperty("user.home");
	    	}
    		
    		// we store the settings into this subfolder
        	settingsDirectory += "/.bbcAnnotator/";
    	} 	
    	
    	// check if directory already exist, else create new directory
    	File dir = new File(settingsDirectory);
    	if (!dir.exists()) {
    	    dir.mkdir();
    	}
    	
		return settingsDirectory;
    }
    
    /**
     * This methods sets some settings, which depends from the current operating system.
     * 
     * @author Christian Henning
     */
    public static void setOSProperties() {
    	// get name of current operating system
    	String operatingSystem = (System.getProperty("os.name")).toUpperCase();
    	
    	if (operatingSystem.contains("MAC")) {
    		// current OS is Mac os
    		// the menubar should be placed with in the taskbar of the operating system
    		System.getProperties().put("apple.laf.useScreenMenuBar", "true");
    	}
    	
    }
}
