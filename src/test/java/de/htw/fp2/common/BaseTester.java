package de.htw.fp2.common;

import org.apache.log4j.Logger;

/**
 * Created by vs on 09.12.14.
 */
public class BaseTester {

    protected Logger log = Logger.getLogger(this.getClass().getCanonicalName());

    protected void printStartTest(String testName){
        log.info("#######################  " +"Start: "+ testName + "  #######################");
    }

    protected void printEndTest(String testName){
        log.info("#######################  " +"End: "+ testName + "  #######################");
    }
}
