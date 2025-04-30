import { Component, OnInit, } from '@angular/core';

@Component({
  selector: 'app-profile',
  templateUrl: './profile.page.html',
  styleUrls: ['./profile.page.scss'],
})
export class ProfilePage implements OnInit {


  async ngOnInit() {
    setTimeout(() => {
      console.log('error');
      throw (new Error('errorror'))
    },
      1000);
  }
}
