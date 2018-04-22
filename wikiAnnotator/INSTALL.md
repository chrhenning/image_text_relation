# Install and Run the wikiAnnotator

The program requires Java 8 and JavaFX 8.

The *wikiAnnotator* is designed to be executed as a JAR file. Hence, one has to construct a JAR file in order to run the program. External libraries have to be included in the JAR file during generation.

External libraries should be placed in the folder `src/lib`. The following libraries are required:

- [hsqldb.jar](http://hsqldb.org/)
- [json-20090211.jar](https://mvnrepository.com/artifact/org.json/json/20090211)
- [javafxsvg.jar](https://mvnrepository.com/artifact/de.codecentric.centerdevice/javafxsvg/1.0.0)

# Usage

The JAR file can be executed from terminal by entering:

```
java -jar wikiAnnotator.jar
```

The dataset has to be manually loaded at the first program start. Therefore, one should use *File -> Open Data Set* and select the fle `articles.jsonl` in the dataset folder. All made changes (entered annotations) will be automatically saved into a local SQL database to prevent data loss. The database is stored in the operating systems application data folder. I.e., after restart of the program on the same machine, the already made annotations will be visible. 

**Note**, this program has not been extensively tested. Therefore, it is recommended to regularly export made annotations (by pressing the *Save As* button) as JSON file.

As the data has been automatically retrieved, there might be invalid samples among the retrieved image-text pairs. For instance, a sample that has no meaningful text as list/tables/equations have been filtered from it. Such samples should be marked as invalid. One can alays go to the *Website* view to compare the retrieved sample with the actual one.

The design of the GUI might be counterintuitive on the first sight. There is a treeview on the left hand sight, that shows all sections of the current article. Only sections that contain images (these are marked with an image icon in the treeview) can be annotated. Note, that some sections have more than one image. In this case, the pagination controls in the image view should be used.

The main pagination control underneath the sample view is used to navigate in between articles.


