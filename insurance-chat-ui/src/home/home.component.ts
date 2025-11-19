import { Component } from '@angular/core';
import { UiServiceService } from 'src/service/ui-service.service';  // ensure correct path

// Message interface
interface Message {
  sender: 'user' | 'bot';
  text: string;
  cached?: boolean;
  source?: string;
}

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {
  messages: Message[] = [];
  userInput: string = '';
  loading: boolean = false;

  constructor(private chatService: UiServiceService) {}

  sendMessage() {
    const query = this.userInput.trim();
    if (!query) return;

    // Push user message to chat window
    this.messages.push({ sender: 'user', text: query });
    this.userInput = '';
    this.loading = true;

    // Send to FastAPI backend
    this.chatService.requestChat({ message: query }).subscribe({
      next: (res) => {
        this.messages.push({
          sender: 'bot',
          text: res.reply || 'No response received.',
          cached: res.cached,
          source: res.source
        });
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.messages.push({
          sender: 'bot',
          text: '⚠️ Error contacting the server.'
        });
        this.loading = false;
      }
    });
  }
}
