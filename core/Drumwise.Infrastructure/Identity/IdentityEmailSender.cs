using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Settings;
using MailKit.Net.Smtp;
using Microsoft.AspNetCore.Identity;
using MimeKit;

namespace Drumwise.Infrastructure.Identity;

public class IdentityEmailSender(IMailSender mailSender) : IEmailSender<ApplicationUser>
{
    public async Task SendConfirmationLinkAsync(ApplicationUser user, string email, string confirmationLink)
    {
        var emailData = new EmailConfirmationMailData(email, confirmationLink);
        await mailSender.SendMailAsync(email, "Confirm your email for Drumwise", emailData).ConfigureAwait(false);
    }

    public Task SendPasswordResetLinkAsync(ApplicationUser user, string email, string resetLink)
    {
        throw new NotImplementedException();
    }

    public Task SendPasswordResetCodeAsync(ApplicationUser user, string email, string resetCode)
    {
        throw new NotImplementedException();
    }
}

public record EmailConfirmationMailData(string EmailAddress, string ConfirmationLink) : IMailData;