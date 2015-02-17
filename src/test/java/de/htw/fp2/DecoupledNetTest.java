package de.htw.fp2;

import de.htw.fp2.common.BaseTester;
import de.htw.fp2.network.DecoupledNet;
import de.htw.fp2.network.classification.DecoupledNetClassification;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by vs on 04.01.15.
 */
public class DecoupledNetTest extends BaseTester {

    @Before
    public void setup() {
        printStartTest(this.getClass().getCanonicalName());
    }

    @After
    public void tearDown() {
        printEndTest(this.getClass().getCanonicalName());
    }

    @Test
    public void testNetwork() {
        DecoupledNetClassification decoupledNetClassification = new DecoupledNetClassification(false, DecoupledNet.Topology.Decoupled_9_18_9);
        assertTrue("Training should run without any Exceptions", decoupledNetClassification.train());
        assertTrue("Test should run without any Exceptions", decoupledNetClassification.test());
    }

}
