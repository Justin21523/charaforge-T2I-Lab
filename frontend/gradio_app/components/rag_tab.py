# frontend/gradio_app/components/rag_tab.py
import gradio as gr
import requests
import json
from pathlib import Path


def create_rag_tab(api_base_url: str = "http://localhost:8000/api/v1"):
    """Create RAG tab for Gradio interface"""

    def upload_document(file, collection_name, chunk_size, chunk_overlap):
        """Upload document to RAG system"""
        try:
            if file is None:
                return "❌ Please select a file", "", ""

            files = {"file": open(file.name, "rb")}
            data = {
                "collection_name": collection_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

            response = requests.post(
                f"{api_base_url}/rag/upload", files=files, data=data
            )

            if response.status_code == 200:
                result = response.json()
                status = f"✅ Uploaded: {result['filename']}\n📄 Chunks: {result['total_chunks']}\n📁 Collection: {result['collection_name']}"
                return status, "", ""
            else:
                return f"❌ Error: {response.text}", "", ""

        except Exception as e:
            return f"❌ Error: {str(e)}", "", ""
        finally:
            if "files" in locals():
                files["file"].close()

    def query_rag(question, collection_name, top_k, max_length, temperature):
        """Query RAG system"""
        try:
            if not question.strip():
                return "❓ Please enter a question", ""

            payload = {
                "question": question,
                "collection_name": collection_name,
                "top_k": top_k,
                "max_length": max_length,
                "temperature": temperature,
            }

            response = requests.post(f"{api_base_url}/rag/ask", json=payload)

            if response.status_code == 200:
                result = response.json()

                answer = f"**Answer:** {result['answer']}\n\n"
                answer += f"**Confidence:** {result['confidence']:.1%}\n\n"
                answer += f"**Sources:** {', '.join(set(result['sources']))}\n\n"

                if result["relevant_chunks"]:
                    answer += "**Relevant Context:**\n"
                    for i, chunk in enumerate(result["relevant_chunks"], 1):
                        answer += f"{i}. {chunk[:200]}...\n\n"

                return answer, ""
            else:
                return f"❌ Error: {response.text}", ""

        except Exception as e:
            return f"❌ Error: {str(e)}", ""

    def get_rag_status():
        """Get RAG system status"""
        try:
            response = requests.get(f"{api_base_url}/rag/status")
            if response.status_code == 200:
                result = response.json()
                status = f"📊 **RAG Status**\n\n"
                status += f"📁 Collections: {', '.join(result['collections']) if result['collections'] else 'None'}\n"
                status += f"📄 Total Documents: {result['total_documents']}\n"
                status += f"🔤 Total Chunks: {result['total_chunks']}\n"
                return status
            else:
                return f"❌ Error: {response.text}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    with gr.Tab("📚 RAG Query"):
        gr.Markdown("## 📚 文件問答 (RAG)")

        with gr.Row():
            with gr.Column(scale=1):
                # Upload section
                gr.Markdown("### 📤 Upload Document")
                upload_file = gr.File(
                    label="📄 Select Document",
                    file_types=[".txt", ".pdf"],
                    type="filepath",
                )

                with gr.Row():
                    collection_name_up = gr.Textbox(
                        label="📁 Collection Name", value="default", max_lines=1
                    )

                with gr.Row():
                    chunk_size = gr.Slider(
                        label="📏 Chunk Size",
                        minimum=128,
                        maximum=1024,
                        value=512,
                        step=64,
                    )
                    chunk_overlap = gr.Slider(
                        label="🔗 Chunk Overlap",
                        minimum=0,
                        maximum=200,
                        value=50,
                        step=10,
                    )

                upload_btn = gr.Button("📤 Upload Document", variant="primary")
                upload_status = gr.Textbox(label="📊 Upload Status", lines=3)

                # Status section
                gr.Markdown("### 📊 System Status")
                status_btn = gr.Button("🔄 Refresh Status")
                status_display = gr.Markdown("Click 'Refresh Status' to view")

            with gr.Column(scale=2):
                # Query section
                gr.Markdown("### ❓ Ask Question")
                question_input = gr.Textbox(
                    label="💬 Your Question",
                    placeholder="What would you like to know about the uploaded documents?",
                    lines=2,
                )

                with gr.Row():
                    collection_name_q = gr.Textbox(
                        label="📁 Collection", value="default", max_lines=1
                    )
                    top_k = gr.Slider(
                        label="🔍 Relevant Chunks",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                    )

                with gr.Row():
                    max_length = gr.Slider(
                        label="📏 Max Answer Length",
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=25,
                    )
                    temperature = gr.Slider(
                        label="🌡️ Creativity",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                    )

                query_btn = gr.Button("❓ Ask Question", variant="primary")
                answer_output = gr.Markdown(label="💡 Answer")

                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        # Event handlers
        upload_btn.click(
            fn=upload_document,
            inputs=[upload_file, collection_name_up, chunk_size, chunk_overlap],
            outputs=[upload_status, question_input, answer_output],
        )

        query_btn.click(
            fn=query_rag,
            inputs=[question_input, collection_name_q, top_k, max_length, temperature],
            outputs=[answer_output, question_input],
        )

        status_btn.click(fn=get_rag_status, outputs=[status_display])

        clear_btn.click(fn=lambda: ("", ""), outputs=[question_input, answer_output])

        # Auto-refresh status on load
        gr.on(triggers=[gr.load()], fn=get_rag_status, outputs=[status_display])
