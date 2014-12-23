package de.htw.fp2.visualization;

import org.apache.log4j.Logger;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Created by patrick on 09.12.14.
 */
public class ImageCreator {

    private static final Logger log = Logger
        .getLogger(ImageCreator.class.getName());

    public static void createGrayScale(File imageFile, double[][] values)
        throws IOException {
        BufferedImage img = new BufferedImage(
            values.length, values[0].length, BufferedImage.TYPE_BYTE_GRAY);
        Color myWhite = new Color(255, 255, 255); // Color white
        int white = myWhite.getRGB();
        for (int x = 0; x < values.length; x++) {
            for (int y = 0; y < values[x].length; y++) {
                img.setRGB(x, y, (int) Math.round(white * values[x][y]));
            }
        }
        ImageIO.write(img, "bmp", imageFile);
    }

    public static void createGrayScale(File imageFile, int[][] values)
        throws IOException {
        BufferedImage img = new BufferedImage(
            values.length, values[0].length, BufferedImage.TYPE_BYTE_GRAY);
        Color myWhite = new Color(255, 255, 255); // Color white
        int white = myWhite.getRGB();
        for (int x = 0; x < values.length; x++) {
            for (int y = 0; y < values[x].length; y++) {
                img.setRGB(x, y, Math.round(white * values[x][y]));
            }
        }
        ImageIO.write(img, "bmp", imageFile);
    }

    public static void createSwitchedGrayScale(File imageFile,
        double[][] values)
        throws IOException {
        BufferedImage img = new BufferedImage(
            values[0].length, values.length, BufferedImage.TYPE_BYTE_GRAY);
        Color myWhite = new Color(255, 255, 255); // Color white
        int white = myWhite.getRGB();
        for (int x = 0; x < values.length; x++) {
            for (int y = 0; y < values[x].length; y++) {
                img.setRGB(y, x, (int) Math.round(white * values[x][y]));
            }
        }
        ImageIO.write(img, "bmp", imageFile);
    }

    public static void drawGradiant(File imageFile, int height)
        throws IOException {
        BufferedImage img = new BufferedImage(
            255, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < 255; x++) {
            Color value = new Color(x, x, x);
            for (int y = 0; y < height; y++) {
                img.setRGB(x, y, Math.round(value.getRGB()));
            }
        }
        ImageIO.write(img, "bmp", imageFile);
    }
}
