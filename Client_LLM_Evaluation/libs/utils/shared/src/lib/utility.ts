/**
 * @param date Date object
 */
export const getTimeDiffInSeconds = (currDate: string, prevDate: string) => {
  const diffInSeconds = (new Date(currDate).getTime() - new Date(prevDate).getTime()) / 1000;
  return diffInSeconds.toFixed(2)
};
