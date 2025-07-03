import React, { useState } from "react";
import axios from "axios";
import "./ChatComponent.css";

const ChatComponent = () => {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {

    if(question.trim() === "") {
      setResponse("Please enter a question.");
      setTimeout(() => {
        setResponse("");
      }, 3000);
      return;
    }

    setResponse("Hygeia is Cooking...");
    setLoading(true);
    const res = await axios.post("http://localhost:8000/api/chat", {
      question: question,
    });
    setResponse(res.data.answer);
    setLoading(false);
    setQuestion("");
  };

  return (
    <div className="chat-container">
      <div className="input-container">
        <div className="query-box">
          <div className="input-box">
            <textarea
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask your medical question..."
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
            />
          </div>
          <div className="send-button" onClick={handleSubmit}>
            {!loading ? (
              <span className="material-symbols-outlined" id="send-icon">
                arrow_upward_alt
              </span>
            ) : (
              <span className="loader"></span>
            )}
          </div>
        </div>
      </div>
      <div className="output-container">
        <div className="response-box">
          <p>{response ? response : "The response will be displayed here!!!"}</p>
        </div>
      </div>
    </div>
  );
};

export default ChatComponent;
