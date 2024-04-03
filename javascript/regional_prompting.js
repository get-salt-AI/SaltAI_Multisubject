import { app } from "/scripts/app.js";

app.registerExtension({
	name: "Comfy.SALT.MultisubjectFaceSwapNode",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "MultisubjectFaceSwapNode") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				this.selected = false
				this.index = 3
                this.serialize_widgets = true;
				
				this.getExtraMenuOptions = function(_, options) {
					options.unshift(
						{
							content: `Add one more face`,
							callback: () => {
								const inputLenth = this.inputs.length - 1;
								this.addInput("face" + inputLenth, "IMAGE");
							},
						},
						{
							content: "Remove last face",
							callback: () => {
								if (this.inputs.length > 3) {
									this.removeInput(-1);
								}
							},
						},
					);
				}

				this.onRemoved = function () {
					// When removing this node we need to remove the input from the DOM
					for (let y in this.widgets) {
						if (this.widgets[y].canvas) {
							this.widgets[y].canvas.remove();
						}
					}
				};
			
				this.onSelected = function () {
					this.selected = true
				}
				this.onDeselected = function () {
					this.selected = false
				}

				return r;
			};
		}
		if (nodeData.name === "RegionalAttentionProcessorNode") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				this.selected = false
				this.index = 3
                this.serialize_widgets = true;
				
				this.getExtraMenuOptions = function(_, options) {
					options.unshift(
						{
							content: `Add one more person`,
							callback: () => {
								let index = Math.floor((this.inputs.length - 4) / 2);
								this.addInput("prompt_embedding_" + index, "CONDITIONING");
								this.addInput("negative_embedding_" + index, "CONDITIONING");
							},
						},
						{
							content: "Remove last person",
							callback: () => {
								if (this.inputs.length <= 2) {
									return;
								}
								this.removeInput(-1);
								this.removeInput(-1);
							},
						},
					);
				}
			
				this.onSelected = function () {
					this.selected = true
				}
				this.onDeselected = function () {
					this.selected = false
				}

				return r;
			};
		}
	},
	loadedGraphNode(node, _) {
		if (node.type === "MultisubjectFaceSwapNode") {
			node.widgets[node.index].options["max"] = node.properties["values"].length-1
		}
		if (node.type === "RegionalAttentionProcessorNode") {
			node.widgets[node.index].options["max"] = node.properties["values"].length-1
		}
	},
});