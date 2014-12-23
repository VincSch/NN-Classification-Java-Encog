package de.htw.fp2.dataset;

import org.json.JSONArray;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.logging.Logger;

/**
 * Created by patrick on 09.12.14.
 */
public class PatternCreator {

    private static final Logger log = Logger
        .getLogger(PatternCreator.class.getName());

    private static final int PATTERN_MAX_COLUMN = 3;
    private static final int PATTERN_MAX_ROW = 3;

    public static List<Pattern> readFrom(File pattern) throws Exception {
        List<Pattern> patterns = new ArrayList<>();
        String fileContent = getFileContent(pattern);
        fileContent = fileContent.replace("\\r\\n", "\n");
        StringTokenizer tokenizer = new StringTokenizer(fileContent, "\n");
        while (tokenizer.hasMoreElements()) {
            Pattern p = parseString(tokenizer.nextToken());
            patterns.add(p);
        }
        return patterns;
    }

    public static Pattern parseString(String patternString) throws Exception {
        boolean[][] value = new boolean[PATTERN_MAX_ROW][PATTERN_MAX_COLUMN];
        JSONArray json = new JSONArray(patternString);
        int length = json.length();
        for (int i = 0; i < length; i++) {
            JSONArray innerArray = json.getJSONArray(i);
            for (int j = 0; j < innerArray.length(); j++) {
                value[i][j] = innerArray.getInt(j) == 0 ? false : true;
            }
        }
        return new Pattern(value);
    }

    private static String getFileContent(File pattern) throws IOException {
        BufferedReader in = new BufferedReader(
            new FileReader(pattern));
        String line = null;
        String content = "";
        while (null != (line = in.readLine())) {
            content += line + "\n";
        }
        return content;
    }

    public static Pattern[] generateRandomPattern(int count) {
        Pattern[] patterns = new Pattern[count];
        for (int i = 0; i < patterns.length; i++) {
            patterns[i] = new Pattern();
            patterns[i].addNoise(18);
        }
        return patterns;
    }
}
