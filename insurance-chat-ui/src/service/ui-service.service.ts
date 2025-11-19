import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class UiServiceService {

  BASE_URL: any;
  constructor(private httpClient: HttpClient) {
    this.BASE_URL = this.determineServerURL();
  }

  private determineServerURL(): string {
    const isLocalhost =
      window.location.hostname === 'localhost' &&
      window.location.port === '4200';
    if (isLocalhost) {
     return 'http://127.0.0.1:5050';
    } else {
      return window.location.origin;
    }
  }

  requestChat(payload: { message: string }): Observable<any> {
    const apiUrl =
      this.BASE_URL + '/chat';
    return this.httpClient.post<any>(apiUrl, payload);
  }
}
