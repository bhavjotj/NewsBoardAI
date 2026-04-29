chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch(() => {});
});

chrome.action.onClicked.addListener((tab) => {
  if (typeof tab.windowId === "number") {
    chrome.sidePanel.open({ windowId: tab.windowId }).catch(() => {});
  }
});
