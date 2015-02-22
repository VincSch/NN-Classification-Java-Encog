package de.htw.fp2;

import de.htw.fp2.common.BaseTester;
import de.htw.fp2.network.DecoupledNet;
import de.htw.fp2.network.classification.DecoupledNetClassification;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
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
    //@Ignore
    public void testBasic_9_3_9() {
        printTestCase("Network: Basic 9 3 9");
        DecoupledNetClassification decoupledNetClassification = new DecoupledNetClassification(DecoupledNet.Topology.Basic_9_3_9);
        assertTrue("Training should run without any Exceptions", decoupledNetClassification.train());
        assertTrue("Test should run without any Exceptions", decoupledNetClassification.test());
    }

    @Test
    //@Ignore
    public void testDecoupled_Linear_9_9_9() {
        printTestCase("Network: Linear Decoupled 9 9 9");
        DecoupledNetClassification decoupledNetClassification = new DecoupledNetClassification(DecoupledNet.Topology.Linear_Decoupled_999);
        assertTrue("Training should run without any Exceptions", decoupledNetClassification.train());
        assertTrue("Test should run without any Exceptions", decoupledNetClassification.test());
    }

    @Test
    //@Ignore
    public void testDecoupled_Linear_With_Pooling_9_9_9_9() {
        printTestCase("Network: Linear Decoupled Pooling 9 9 9 9");
        DecoupledNetClassification decoupledNetClassification = new DecoupledNetClassification(DecoupledNet.Topology.Linear_Pooling_Decoupled_999);
        assertTrue("Training should run without any Exceptions", decoupledNetClassification.train());
        assertTrue("Test should run without any Exceptions", decoupledNetClassification.test());
    }

    @Test
    //@Ignore
    public void testDecoupled_9_18_9() {
        printTestCase("Network: Linear Decoupled 9 18 9");
        DecoupledNetClassification decoupledNetClassification = new DecoupledNetClassification(DecoupledNet.Topology.ES_Decoupled_9_18_9);
        assertTrue("Training should run without any Exceptions", decoupledNetClassification.train());
        assertTrue("Test should run without any Exceptions", decoupledNetClassification.test());
    }

    @Test
    //@Ignore
    public void testDecoupled_With_Pooling_9_18_9() {
        printTestCase("Network: Linear Decoupled Pooling 9 18 9");
        DecoupledNetClassification decoupledNetClassification = new DecoupledNetClassification(DecoupledNet.Topology.ES_Pooling_Decoupled_9_18_9);
        assertTrue("Training should run without any Exceptions", decoupledNetClassification.train());
        assertTrue("Test should run without any Exceptions", decoupledNetClassification.test());
    }

    @Test
    @Ignore
    public void testDecoupled_Random_9_18_9() {
        printTestCase("Network: Linear Decoupled Pooling 9 18 9");
        DecoupledNetClassification decoupledNetClassification = new DecoupledNetClassification(DecoupledNet.Topology.RandomDecoupled_9_18_9);
        assertTrue("Training should run without any Exceptions", decoupledNetClassification.train());
        assertTrue("Test should run without any Exceptions", decoupledNetClassification.test());
    }

}
