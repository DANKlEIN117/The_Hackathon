"""
Fixed Hugging Face Web Chatbot - Compatible with Latest Transformers
No import errors, ready to run!
"""

# Install these packages:
# pip install transformers torch gradio

import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

class WebChatbot:
    def __init__(self):
        """Initialize the chatbot - handles both old and new transformers versions"""
        print("🤖 Loading chatbot model...")
        
        try:
            # Method 1: Try using the simple conversational pipeline
            self.chatbot = pipeline(
                "conversational",
                model="microsoft/DialoGPT-medium",
                device=-1  # CPU
            )
            self.method = "pipeline"
            print("✅ Using conversational pipeline")
            
        except Exception as e:
            print(f"Pipeline method failed: {e}")
            print("🔄 Trying alternative method...")
            
            # Method 2: Direct model loading (more compatible)
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.method = "direct"
            print("✅ Using direct model loading")
        
        print("✅ Chatbot loaded successfully!")
    
    def generate_response_pipeline(self, message, history):
        """Generate response using pipeline method"""
        try:
            # Convert history to the format expected by pipeline
            conversation_history = []
            for human, assistant in history:
                conversation_history.extend([human, assistant])
            conversation_history.append(message)
            
            # Join conversation
            input_text = " ".join(conversation_history[-10:])  # Keep last 10 exchanges
            
            # Create a simple conversation object
            class SimpleConversation:
                def __init__(self):
                    self.past_user_inputs = []
                    self.generated_responses = []
                
                def add_user_input(self, text):
                    self.past_user_inputs.append(text)
                
                def mark_processed(self):
                    pass
                
                def append_response(self, text):
                    self.generated_responses.append(text)
            
            conv = SimpleConversation()
            
            # Add history
            for i, (human, assistant) in enumerate(history):
                conv.add_user_input(human)
                conv.append_response(assistant)
            
            # Add current message
            conv.add_user_input(message)
            
            # Generate response
            result = self.chatbot(conv)
            return result.generated_responses[-1]
            
        except Exception as e:
            return f"Error with pipeline method: {str(e)}"
    
    def generate_response_direct(self, message, history):
        """Generate response using direct model method"""
        try:
            # Build conversation context
            conversation_text = ""
            
            # Add history
            for human, assistant in history[-5:]:  # Keep last 5 exchanges
                conversation_text += f"Human: {human}{self.tokenizer.eos_token}"
                conversation_text += f"Bot: {assistant}{self.tokenizer.eos_token}"
            
            # Add current message
            conversation_text += f"Human: {message}{self.tokenizer.eos_token}Bot:"
            
            # Tokenize
            inputs = self.tokenizer.encode(
                conversation_text, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the bot's response
            bot_response = full_response[len(conversation_text):].strip()
            
            # Clean up response
            if bot_response.startswith("Bot:"):
                bot_response = bot_response[4:].strip()
            
            return bot_response if bot_response else "I'm not sure how to respond to that. Could you try rephrasing?"
            
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again!"
    
    def chat_response(self, message, history):
        """Main response function that tries both methods"""
        if not message.strip():
            return "Please enter a message!"
        
        if self.method == "pipeline":
            response = self.generate_response_pipeline(message, history)
        else:
            response = self.generate_response_direct(message, history)
        
        return response

def create_chatbot_interface():
    """Create and configure the Gradio interface"""
    
    # Initialize chatbot
    bot = WebChatbot()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    .chat-message {
        font-size: 16px !important;
    }
    .message-row {
        margin: 8px 0 !important;
    }
    footer {
        visibility: hidden;
    }
    """
    
    # Create the interface
    with gr.Blocks(
        css=custom_css,
        title="🤖 AI Chatbot",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate"
        )
    ) as interface:
        
        # Header section
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="text-align: center; padding: 20px;">
                    <h1 style="color: #2563eb; margin-bottom: 10px;">🤖 AI Chatbot</h1>
                    <p style="color: #64748b; font-size: 18px;">Powered by Hugging Face Transformers</p>
                    <p style="color: #94a3b8;">Ask me anything! I can help with questions, have conversations, or just chat.</p>
                </div>
                """)
        
        # Main chat interface
        chatbot_interface = gr.ChatInterface(
            fn=bot.chat_response,
            chatbot=gr.Chatbot(
                height=500,
                show_label=False,
                container=True,
                bubble_full_width=False
            ),
            textbox=gr.Textbox(
                placeholder="Type your message here... Press Enter to send",
                container=False,
                scale=7
            ),
            submit_btn="Send 💬",
            retry_btn="🔄 Retry",
            undo_btn="↩️ Undo",
            clear_btn="🗑️ Clear Chat",
            examples=[
                " Hello! How are you today?",
                " What's your favorite topic to discuss?",
                " Can you tell me a funny joke?",
                " Recommend a good book to read",
                " What kind of music do you like?",
                " Tell me something interesting about space",
                " How does artificial intelligence work?",
                " What's your favorite food?"
            ]
        )
        
        # Footer with usage stats and tips
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: #f8fafc; border-radius: 8px; margin-top: 20px;">
                    <h3 style="color: #475569; margin-bottom: 10px;">💡 Tips for Better Conversations</h3>
                    <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                        <div style="margin: 5px; padding: 10px; background: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <strong>📝 Be Specific</strong><br>
                            <small>Ask detailed questions for better responses</small>
                        </div>
                        <div style="margin: 5px; padding: 10px; background: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <strong>🧠 Context Matters</strong><br>
                            <small>I remember our conversation history</small>
                        </div>
                        <div style="margin: 5px; padding: 10px; background: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <strong>🎯 Try Examples</strong><br>
                            <small>Click the example prompts above to start</small>
                        </div>
                    </div>
                </div>
                """)
    
    return interface

def main():
    """Main execution function"""
    print(" Starting Hugging Face Web Chatbot...")
    print(" Loading required components...")
    
    try:
        # Create the interface
        interface = create_chatbot_interface()
        
        print("\n" + "="*60)
        print(" WEB CHATBOT READY!")
        print(" Opening in your browser...")
        print(" Interface will be available at: http://127.0.0.1:7860")
        print(" Want to share? Change share=False to share=True in the code")
        print(" Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Launch the interface
        interface.launch(
            share=False,  # Change to True for public sharing
            inbrowser=True,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n Chatbot stopped. Thanks for using it!")
    except ImportError as e:
        print(f" Missing package: {e}")
        print(" Please install: pip install transformers torch gradio")
    except Exception as e:
        print(f" Error: {str(e)}")
        print("\n Troubleshooting steps:")
        print("1. pip install --upgrade transformers torch gradio")
        print("2. Make sure you have internet connection for model download")
        print("3. Try restarting your Python environment")

if __name__ == "__main__":
    main()