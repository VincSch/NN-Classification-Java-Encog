package de.htw.fp2;

import org.apache.log4j.Logger;
import org.junit.Test;

/**
 * Created by vs on 09.12.14.
 */
public class XORTest {

    private static Logger log = Logger.getLogger(XORTest.class.getName());

    @Test
    public void testNetwork() {
        SimpleXOR xor = new SimpleXOR();
        xor.runXOR();
        log.info("========Test run Done ==========");
    }
}
