using System.Security.Claims;
using Ardalis.GuardClauses;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Infrastructure.Identity.Constants;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Identity;

namespace Drumwise.Infrastructure.Identity;

public class CurrentUser(IHttpContextAccessor httpContextAccessor, IIdentityService identityService)
    : IUser
{
    public string? Id => httpContextAccessor.HttpContext?.User.FindFirstValue(ClaimTypes.NameIdentifier);

    public async Task<string?> GetUserRole()
    {
        var role = await identityService.GetUserRole(Id);
        Guard.Against.NullOrEmpty(role, nameof(role));

        return role;
    }

    public async Task<bool> IsInRoleAsync(string role)
    {
        return await identityService.IsInRoleAsync(Id, role);
    }
}