import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "Diskfo.TimerDisplay",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ExecutionTimerEnd") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				if (message?.text) {
					// יצירת ווידג'ט להצגת טקסט אם הוא לא קיים
					let w = this.widgets?.find((w) => w.name === "display_text");
					if (!w) {
						w = this.addWidget("text", "display_text", "", () => {});
						w.disabled = true;
					}
					w.value = "🕒 " + message.text[0];
					this.onResize?.(this.size);
				}
			};
		}
	},
});
