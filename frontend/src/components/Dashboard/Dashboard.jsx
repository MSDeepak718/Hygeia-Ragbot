import ChatComponent from '../ChatComponent/ChatComponent'
import './Dashboard.css';


const Dashboard = () => {
  return (
    <div className='main-container'>
        <div className="header">
            <h1>Hygeia-Ragbot</h1>
        </div>
        <div className="chat-component">
            <ChatComponent/>
        </div>
    </div>
  )
}

export default Dashboard