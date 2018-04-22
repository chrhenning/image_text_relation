# Install and Run the bbcAnnotator

The program requires Java 8 and JavaFX 8.

The *bbcAnnotator* is designed to be executed as a JAR file. Hence, one has to construct a JAR file in order to run the program. A messy detail in the implementation is, that the dataset has to be inside the JAR file itself as well as the external libraries.

Hence, before generating the JAR file, one has to download the [dataset](http://homepages.inf.ed.ac.uk/s0677528/data.html). The test samples should be placed inside the folder `src/data/data_test` and the training files in the folder `src/data/data_train`, respectively.

External libraries should be placed in the folder `src/lib`. The following libraries are required:

- [hsqldb.jar](http://hsqldb.org/)
- [json-20090211.jar](https://mvnrepository.com/artifact/org.json/json/20090211)

# Usage

The JAR file can be executed from terminal by entering:

```
java -jar bbcAnnotator.jar
```

The dataset is automatically loaded. All made changes (entered annotations) will be automatically saved into a local SQL database to prevent data loss. The database is stored in the operating systems application data folder. I.e., after restart of the program on the same machine, the already made annotations will be visible. 

**Note**, this program has not been extensively tested. Therefore, it is recommended to regularly export made annotations (by pressing the *Save As* button) as JSON file.


