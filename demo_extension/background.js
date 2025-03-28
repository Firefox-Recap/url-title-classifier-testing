// Function to ensure the ML engine is ready
async function ensureEngineIsReady(tabId) {
  const { engineCreated } = await browser.storage.session.get({
    engineCreated: false,
  });

  if (engineCreated) {
    return;
  }

  const listener = (progressData) => {
    browser.tabs.sendMessage(tabId, { type: "progress", data: progressData });
  };
  browser.trial.ml.onProgress.addListener(listener);

  try {
    await displayMessage(tabId, "Initializing ML model...");
    await browser.trial.ml.createEngine({
      modelHub: "huggingface",
      taskName: "text-classification",
      modelId: "firefoxrecap/URL-TITLE-classifier",
      dtype: "q8",
    });
    browser.storage.session.set({ engineCreated: true });
    await displayMessage(tabId, "ML model ready!");
  } catch (err) {
    await displayMessage(
      tabId,
      `Failed to initialize ML model: ${err.message}`
    );
    throw err;
  } finally {
    browser.trial.ml.onProgress.removeListener(listener);
  }
}
// Function to classify a URL and title
async function classifyURLAndTitle(url, title, tabId) {
  try {
    // Combine URL and title for classification
    const textToClassify = `${url}: ${title}`;
    console.log("Classifying:", textToClassify);

    // Ensure the ML engine is ready before running inference
    await ensureEngineIsReady(tabId);

    // Run the engine with the input text
    const result = await browser.trial.ml.runEngine({
      args: [textToClassify],
    });

    console.log("Classification result:", result);
    await displayMessage(
      tabId,
      `Classified as: ${result[0].label} (${(result[0].score * 100).toFixed(
        2
      )}%)`
    );

    return result;
  } catch (error) {
    console.error("Classification error:", error);
    await displayMessage(tabId, `Error: ${error.message}`);
    return null;
  }
}

// Listen for tab updates to classify URLs as they're visited
browser.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (
    changeInfo.status === "complete" &&
    tab.url &&
    !tab.url.startsWith("moz-extension://")
  ) {
    try {
      await classifyURLAndTitle(tab.url, tab.title || "", tabId);
    } catch (error) {
      console.error(error);
    }
  }
});

// Function to handle manual classification requests
browser.runtime.onMessage.addListener(async (message, sender) => {
  if (message.action === "classify") {
    const tabId = sender.tab ? sender.tab.id : null;
    if (tabId) {
      return classifyURLAndTitle(message.url, message.title, tabId);
    }
  }
  return false;
});

// Example classification Run a test classification on startup 
async function runTestClassification() {
  const testText =
    "https://www.techhealth.gov/articles/new-technology-for-Travel: Government Announces New Technology for Travel Innovation";
  console.log("Running test classification with:", testText);

  try {
    // Use the active tab for displaying messages
    const tabs = await browser.tabs.query({
      active: true,
      currentWindow: true,
    });
    if (tabs.length > 0) {
      const tabId = tabs[0].id;

      // Ensure engine is ready and run classification
      await ensureEngineIsReady(tabId);
      const result = await browser.trial.ml.runEngine({
        args: [testText],
      });

      console.log("Test classification result:", result);
    }
  } catch (error) {
    console.error("Test classification error:", error);
  }
}

// Uncomment to run the test classification on startup
//runTestClassification();
