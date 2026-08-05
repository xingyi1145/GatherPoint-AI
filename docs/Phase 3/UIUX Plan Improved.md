# GatherPoint AI: UI/UX Revamp Plan

## 1. The Great Purge (Decluttering)
To achieve a clean, minimalist interface, we will actively remove all instructional placeholders and debugging panels.

* **Delete the Right Sidebar:** The entire "Live Plan / Decision context / Memory retrieval" column will be removed. 
* **Remove Placeholder Descriptions:** Delete the "Manage the meetup group...", "Profiles will appear here...", and "Ready to coordinate..." text blocks entirely.
* **Remove Cluttering Emojis:** Strip out the unnecessary emojis next to headers to elevate the formality and cleanliness of the design.
* **Simplify the Header:** Move the main branding to the top left of the main chat column. It will simply read: 
    * **GatherPoint**
    * *Local-first group coordination powered by AMD ROCm & vLLM.*

## 2. The Gemini-Style Layout
The app will be restructured into two distinct columns: a functional left sidebar and a wide, distraction-free main chat interface.

* **The Left Sidebar (System & Context):**
    * **Active Group:** A clean list of current participants (e.g., Alice, Bob).
    * **Friend Profiles:** Minimalist tags displaying their constraints (e.g., *Vegan*, *Transit*).
    * **System Status:** A small, sleek indicator at the bottom corner showing "vLLM: Online".
* **The Main Chat Area:**
    * **Bottom-Anchored Input:** The chat input box will sit at the bottom of the screen, just like modern LLM interfaces.
    * **Clean Message Bubbles:** User and Assistant messages will alternate with distinct but subtle background colors (configured in `.streamlit/config.toml`).

## 3. Dynamic Visualizations (Show, Don't Tell)
Instead of dumping raw data into the chat, the agent will render rich, interactive components.

* **The "Thinking" Expander:** We will use `st.status("Routing via AMD Radeon GPU...", expanded=False)` to hide the messy ReAct loop logs. The user only sees a clean spinner, but technical judges can click to expand and see the raw tool execution.
* **Interactive Mapping:** When the agent finalizes a location, we will extract the latitude and longitude and inject `st.map(data)` directly into the final assistant chat bubble, providing an immediate visual payload.

## 4. Technical Implementation Notes
* Update `st.set_page_config()` to use `layout="wide"` and `initial_sidebar_state="expanded"` to ensure the UI spans the screen correctly and the left sidebar is always visible on boot.
* Use Streamlit's native `st.chat_message()` and `st.chat_input()` to handle the core conversational loop seamlessly.