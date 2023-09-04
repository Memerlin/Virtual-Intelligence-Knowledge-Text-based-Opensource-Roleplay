# Virtual Intelligence Knowledge Text-based Opensource Roleplay
 Or how I like to call it, Viktor, is my attempt to training AI from scratch. The training data might be released someday, I'm a bit uncomfortable letting people see my chatlogs.
 Special thanks to OpenAI, Claude, HuggingFace, Riot Games and my friends at Uni.

# So, how am I supossed to use this?
That's the part that I'm still figuring out.
1. Right now, you just make a column called "text" on an excel/google sheets file and put all the data you want under a column.
2. Export as .csv
3. Convert it to .JSONl (don't ask me why).
4. Save the file
5. Pass it through preprocess_and_training.py
6. Wait an eternity if you're like me and don't have a GPU.
7. Load it with loading_and_chatting.py
8. Have fun.
Right now its just a simple transformers model. Hopefully I'll figure out how to make it better.
