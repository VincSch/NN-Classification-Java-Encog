package de.htw.fp2;

import de.htw.fp2.common.BaseTester;
import de.htw.fp2.examples.SimpleXOR;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by vs on 09.12.14.
 */
public class XORTest extends BaseTester {

    @Before
    public void setup(){
        printStartTest(this.getClass().getCanonicalName());
    }

    @After
    public void tearDown(){
        printEndTest(this.getClass().getCanonicalName());
    }

    @Test
    public void testNetwork() {
        SimpleXOR xor = new SimpleXOR();
        assertTrue("Training should run without any Exceptions", xor.runXOR());
    }
}
