using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;

namespace Drumwise.Infrastructure.Identity;

public class IdentityService(UserManager<ApplicationUser> userManager,
        IUserClaimsPrincipalFactory<ApplicationUser> userClaimsPrincipalFactory,
        IAuthorizationService authorizationService)
    : IIdentityService
{
    public  async Task<string?> GetUserNameAsync(string userId)
    {
        var user = await userManager.Users.FirstAsync(u => u.Id == userId);
        return user.UserName;
    }

    public async Task<bool> IsInRoleAsync(string userId, string role)
    {
        var user = userManager.Users.SingleOrDefault(u => u.Id == userId);
        return user != null && await userManager.IsInRoleAsync(user, role);
    }

    public async Task<bool> AuthorizeAsync(string userId, string policy)
    {
        var user = userManager.Users.SingleOrDefault(u => u.Id == userId);
        if (user is null)
            return false;

        var principal = await userClaimsPrincipalFactory.CreateAsync(user);
        var result = await authorizationService
            .AuthorizeAsync(principal, policy);

        return result.Succeeded;
    }

    public async Task<(Result Result, string UserId)> CreateUserAsync(ApplicationUserDto applicationUserDto)
    {
        var user = new ApplicationUser()
        {
            UserName = applicationUserDto.Username,
            Name = applicationUserDto.Name,
            Surname = applicationUserDto.Surname,
            Email = applicationUserDto.Email
        };

        var result = await userManager.CreateAsync(user, applicationUserDto.Password);
        return (result.ToApplicationResult(), user.Id);
    }

    public async Task<Result> DeleteUserAsync(string userId)
    {
        var user = userManager.Users.SingleOrDefault(u => u.Id == userId);
        return user != null ? await PerformDeleteUser(user) : Result.Success();
    }

    private async Task<Result> PerformDeleteUser(ApplicationUser user)
    {
        var result = await userManager.DeleteAsync(user);
        return result.ToApplicationResult();
    }
}