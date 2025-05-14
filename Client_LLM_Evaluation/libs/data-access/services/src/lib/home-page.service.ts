import { inject, Injectable } from '@angular/core';
import { HttpClientService } from './common/http-client.service';
import { Observable, Subject } from 'rxjs';
import { HomeSlider } from '@ngtx-apps/data-access/models';

@Injectable({
  providedIn: 'root',
})
export class HomePageService {
  private readonly httpClientService = inject(HttpClientService);

  public getHomeSlider(): HomeSlider[] {
    const homeSlider: HomeSlider[] = [
      {
        ImageURL: 'https://exdevapi.qatarairways.com.qa/pic/400k_final.jpg',
        VideoSrc: '',
        VideoType: '',
        Header: 'Celebrating  400K student club members',
        SubHeader: 'Stay tuned as we continue to enhance Student club',
      },
      {
        ImageURL: '',
        VideoSrc:
          'https://exdevapi.qatarairways.com.qa/vid/QR_X_PSG_REVEAL_PROMO.mp4',
        VideoType: 'video/mp4',
        Header: 'Official Jersey Partner of PSG',
        SubHeader: 'We enter a special era with Les Rouge-et-Bleu',
      },
      {
        ImageURL: '',
        VideoSrc:
          'https://exdevapi.qatarairways.com.qa/vid/37075_Summer-in-Qatar-Video_v5.mp4',
        VideoType: 'video/mp4',
        Header: 'Summer In Qatar',
        SubHeader: 'Get ready for a truly wonderful family time',
      },
      {
        ImageURL: 'assets/ROC17QTA-6-2.jpg',
        VideoSrc: '',
        VideoType: '',
        Header: 'World Bicycle Day on 03 Jun 2022',
        SubHeader: 'MOPH event on bicycle day in Qatar',
      },
      {
        ImageURL: 'assets/OnlyOneEarth2022.jpg',
        VideoSrc: '',
        VideoType: '',
        Header: 'Environment Dat on 03 Jun 2022',
        SubHeader: 'Qatar celebrates Environment Day',
      },
      {
        ImageURL: 'https://exdevapi.qatarairways.com.qa/pic/HBR-Banner.jpg',
        VideoSrc: '',
        VideoType: '',
        Header: 'Harvard Business Review Magazine',
        SubHeader: 'Enjoy complimentary access',
      },
    ];

    return homeSlider;
  }

  public async addUser(data: string): Promise<any> {
    return await this.httpClientService.post(
      'https://reqbin.com/echo/post/json',
      data
    );
  }

  public getUserO(): Observable<any> {
    const subject: Subject<any> = new Subject();
    this.httpClientService.get('https://randomuser.me/api').subscribe(subject);
    return subject;
  }

  public async getUser(): Promise<any> {
    return 'User';
  }
}
